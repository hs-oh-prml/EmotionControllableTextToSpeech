import numpy as np 
import fastwer 
import os 

f = open("./result/gt_result.txt", 'r')
# f = open("./result/fastspeech_result.txt", 'r')
# f = open("./result/gst_result.txt", 'r')

a_err_list = [] 
d_err_list = [] 
f_err_list = [] 
h_err_list = [] 
n_err_list = [] 
sa_err_list = [] 
su_err_list = [] 

temp_err = []
cur_emo = None


while True:
    line = f.readline()
    if not line: break
    items = line.split("|")
    text = items[1]
    # GT
    items = items[0].replace(".wav", "").split("_")
    emo = items[1]
    f_name = "{}_{}_{}".format(items[0], items[1], items[2])

    # GST
    # items = items[0].replace(".wav", "").replace(".npy", "").split("_")
    # emo = items[2]
    # f_name = "{}_{}_{}".format(items[1], items[2], items[3])

    # FS
    # items = items[0].replace(".wav", "").replace(".npy", "").split("_")
    # emo = items[3]
    # f_name = "{}_{}_{}".format(items[2], items[3], items[4])

    # n = np.load("/hd0/hs_oh/emotion_korea/{}/{}.npz".format(emo, f_name), 'r')
    f_path = "/hd0/cb_im/emotion_kor/{}_3000_16000Hz/txt/{}.txt".format(emo, f_name)
    if not os.path.exists(f_path): 
        print("NO FILE: {}".format(f_path))
        continue
    n = open(f_path, 'r')

    # gt = str(n["text"])
    gt = n.readline()
    gt = gt.replace(" ", "")
    gt = gt.replace(".", "")

    hypo = text.strip()
    hypo = hypo.replace(" ", "")
    hypo = hypo.replace(".", "")
    print(gt)
    print(hypo)
    cer = fastwer.score([hypo], [gt], char_level=True)
    if emo == "ang":
        a_err_list.append(cer)
    if emo == "dis":
        d_err_list.append(cer)
    if emo == "fea":
        f_err_list.append(cer)
    if emo == "neu":
        n_err_list.append(cer)
    if emo == "hap":
        h_err_list.append(cer)
    if emo == "sad":
        sa_err_list.append(cer)        
    if emo == "sur":
        su_err_list.append(cer)
    n.close()

    # wer = fastwer.score([hypo], [gt])

    # print(gt)
    # print(hypo)
    # print(cer)
    # err_list.append(cer)
    # print(wer)
a_err = np.mean(a_err_list)
print(len(a_err_list), "ang", a_err)

d_err = np.mean(d_err_list)
print(len(d_err_list), "dis", d_err)

f_err = np.mean(f_err_list)
print(len(f_err_list), "fea", f_err)

n_err = np.mean(n_err_list)
print(len(n_err_list), "neu", n_err)

h_err = np.mean(h_err_list)
print(len(h_err_list), "hap", h_err)

sa_err = np.mean(sa_err_list)
print(len(sa_err_list), "sad", sa_err)

su_err = np.mean(su_err_list)
print(len(su_err_list), "sur", su_err)

# mean_err = np.mean(err_list)
# print("CER: {}".format(mean_err))