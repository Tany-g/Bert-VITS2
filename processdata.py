import os

# 定义目标语言和名称列表
names = [('keli', 'ZH'), 
         ('kl_dal', 'ZH'),
         ('zaoyou','ZH')]

# 输出文件路径
out_file = "filelists/genshin.list"

def process():
    with open(out_file, 'w', encoding="utf-8") as wf:
        for item in names:
            ch_name = item[0]
            ch_language = item[1]
            path = f"./raw/{ch_name}"
            files = os.listdir(path)
            
            for f in files:
                if f.endswith('.lab'):
                    with open(os.path.join(path, f), 'r', encoding="utf-8") as perFile:
                        line = perFile.readline()
                        result = f"./dataset/{ch_name}/{f.split('.')[0]}.wav|{ch_name}|{ch_language}|{line}"
                        wf.write(f'{result}\n')

if __name__ == '__main__':
    process()