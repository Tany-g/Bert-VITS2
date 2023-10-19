from fastapi import FastAPI, Query, HTTPException, Response
from io import BytesIO
import torch
from av import open as avopen

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile

app = FastAPI()
def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str,dev)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(dev).unsqueeze(0)
        tones = tones.to(dev).unsqueeze(0)
        lang_ids = lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        ja_bert = ja_bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        return audio

def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text

def wav2(i, o, format):
    inp = avopen(i, "rb")
    out = avopen(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()

# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

dev = "cuda"
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(dev)
_ = net_g.eval()

_ = utils.load_checkpoint("logs/genshin/G_60000.pth", net_g, None, skip_optimizer=True)

@app.get("/")
def main(
    speaker: str = Query("sllh", description="Speaker ID"),
    text: str = Query("吃葡萄不吐葡萄皮", description="Text to synthesize"),
    sdp_ratio: float = Query(0.2, description="SDP Ratio"),
    noise: float = Query(0.5, description="Noise Scale"),
    noisew: float = Query(0.6, description="Noise Scale W"),
    length: float = Query(1.2, description="Length Scale"),
    language: str = Query("ZH", description="Language (ZH or JA)"),
    format: str = Query("wav", description="Output audio format (wav, mp3, ogg)")
):
    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language,
        )

    with BytesIO() as wav:
        wavfile.write(wav, hps.data.sampling_rate, audio)
        torch.cuda.empty_cache()
        if format == "wav":
            return Response(wav.getvalue(), media_type="audio/wav")
        wav.seek(0, 0)
        with BytesIO() as ofp:
            wav2(wav, ofp, format)
            return Response(
                ofp.getvalue(), media_type="audio/mpeg" if format == "mp3" else "audio/ogg"
            )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8889)
