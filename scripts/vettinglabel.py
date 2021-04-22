import tkinter
import os
import pickle
import functools
import sys
​
from PIL import Image, ImageTk
​
​
def load_imgs(inlist, indir, wk=None):
    
    with open(inlist, 'r') as f:
        fltr = f.read().splitlines()
​
    fs = [os.path.join(indir, f) for f in fltr]
​
    return fs
​
​
def load():
    try:
        with open('labels.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return [{},{},{}]
​
​
@functools.lru_cache(maxsize=200)
def loadim(f, w):
    im = Image.open(f)
    maxw = w.winfo_screenwidth()
    maxh = w.winfo_screenheight() - 160
    prop = min(1.0, maxw / im.width, maxh / im.height)
    if prop < 1.0:
        im.thumbnail((im.width * prop, im.height * prop), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image=im), im
​
​
def parseargs():
    user_str = ''
    user_id = 0
    indir = ''
    args = sys.argv[1:]
    if len(args) > 0:
        inlist = args[0]
    if len(args) > 1:
        indir = args[1]
    if len(args) > 2:
        user_str = args[2]
    if len(args) > 3:
        user_id = args[3]
    return inlist, indir, user_str, user_id
​
​
def main_app():
    inlist, indir, usr, uid = parseargs()
​
    fs = load_imgs(inlist, indir)
    i = 0
    j = 1
    pages = 3
    lbl = load()
​
    while (os.path.basename(fs[i]) in lbl[0]) and (os.path.basename(fs[i]) in lbl[1]):
        i += 1
        if i >= len(fs):
            i = 0
            break
​
    searchmode = False
    search_text = ''
​
    def fw():
        nonlocal i,j
        i += 1
        i %= len(fs)
        j = 1
        update(j)
​
    def bk():
        nonlocal i,j
        i -= 1
        i %= len(fs)
        j = 1
        update(j)
​
    def fpg():
        nonlocal j
        j += 1
        if j>pages:
            j %= pages
        update(j)
    
    def bpg():
        nonlocal j
        j -= 1
        if j==0:
            j = pages
        update(j)
​
​
​
​
    def save():
        fname = 'labels.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(lbl, f, pickle.HIGHEST_PROTOCOL)
            set_status(f'saved to {fname}')
            return f.name
​
    def export():
        if usr:
            fname = f'{usr}dispositions.csv'
        else:
            fname = 'dispositions.csv'
        with open(fname, 'wt') as f:
            for k in fs:
                k = os.path.basename(k)
                if (k in lbl[0]) and (k in lbl[1]):
                    lbls = [l.get(k, '') for l in lbl]
                    f.write(f'{k},{uid},{",".join(lbls)}\n')
                
            set_status(f'exported to {fname}')
            return f.name
​
    def label_str():
        if searchmode:
            sf = [f for f in fs if os.path.basename(f).startswith(search_text)]
            n = len(sf)
            return f'Search: {search_text} ({n} matches)'
        k = os.path.basename(fs[i])
        lbls = [l.get(k, '').upper() for l in lbl]
        lbltext = ','.join(lbls[:2])
        if lbls[2]:
            lbltext += ' / S'
        return lbltext
​
    def status_str():
        return f'{i + 1} / {len(fs)} ({min(len(lbl[0]), len(lbl[1]))} labels)'
​
    w = tkinter.Tk()
    w.tk.call('tk', 'scaling', 2.0)
​
    tki, im = loadim(fs[i]+".page%d.png" % j, w)
    pl = tkinter.Label(w, image=tki, width=im.width, height=im.height)
    pl.pack()
​
    frame = tkinter.Frame(w, width=80, height=1)
    frame.pack()
​
    txts = tkinter.Label(frame, text=status_str(), width=40, height=1)
    txts.config(font=("Arial", 14))
    txts.config(fg="blue")
    txts.pack(side=tkinter.LEFT)
​
    txt = tkinter.Label(frame, text=label_str(), width=40, height=1)
    txt.config(font=("Arial", 14))
    txt.config(fg="green")
    txt.pack(side=tkinter.LEFT)
​
    txtm = tkinter.Label(frame, text='', width=40, height=1)
    txtm.pack(side=tkinter.LEFT)
​
    def set_status(msg):
        txtm.configure(text=msg)
        w.update()
​
    def update(pageindex):
        set_status('loading')
        update_text()
        tki, _ = loadim(fs[i]+".Page%d.png" % pageindex, w)
        pl.configure(image=tki)
        pl.image = tki
        set_status('')
​
    def update_text():
        txt.configure(text=label_str())
        txts.configure(text=status_str())
​
​
​
    def keydown(e):
        nonlocal searchmode, search_text, i
        if searchmode:
            if e.keycode == 3342463:
                search_text = search_text[:-1]
            elif e.keycode == 3473435:
                searchmode = False
            elif e.keycode == 2359309:
                sf = [f for f in fs if os.path.basename(f).startswith(search_text)]
                i = fs.index(sf[0])
                searchmode = False
                update(1)
            elif e.char >= '0' and e.char <= '9':
                search_text += e.char
            update_text()
            return
​
        fn = os.path.basename(fs[i])
        if e.char == 'q':
            save()
            w.quit()
        elif e.char == 'w':
            save()
        elif e.char == 'x':
            export()
        elif e.char in ('e', 'p', 'n'):
            lbl[0][fn] = e.char
            update_text()
            w.update()
            if fn in lbl[1]:
                fw()
        elif e.char in ('t', 'b', 'u'):
            lbl[1][fn] = e.char
            update_text()
            w.update()     
            if fn in lbl[0]:
                fw()
        elif e.char == 's':
            if fn not in lbl[2]:
                lbl[2][fn] = 's'
            else:
                del lbl[2][fn]
            update_text()
            w.update()     
            if fn in lbl[0] and fn in lbl[1]:
                fw()
​
        elif e.char == '/':
            searchmode = True
            search_text = ''
            update_text()
​
        elif e.keycode == 3342463:
            if fn in lbl[0]:
                del lbl[0][fn]
            if fn in lbl[1]:
                del lbl[1][fn]
            if fn in lbl[2]:
                del lbl[2][fn]
            update_text()
            w.update()     
​
        elif e.keycode == 8189699:
            fw()
        elif e.keycode == 8124162:
            bk()
        elif e.keycode == 8255233:
            fpg()
        elif e.keycode == 8320768:
            bpg()
​
        else:
            print(e.char, e.keycode)
​
​
    w.bind("<Key>", keydown)
​
    w.mainloop()
​
    unk = []
    msg = ''
​
​
def main():
    helptxt = """
    Usage:
      python3 label.py <week no> <user initials> <user id>
​
    Keys:
      Arrow keys - navigate, up-down, cycle through the pages from one tic, left, right go to the previous/next tic
      q - save labels and quit
      w - save labels
      x - export to CSV
      p, e, n - set first label (Planet, Eclipsing binary, uNdecided) 
      t, b, u - set second label (on Target, Background, Undecided)
        when both the first and the second labels are set, the app moves forward
      s - toggle third label (Single transit)
      del - clear all labels
      / - search
          <Enter> select first search match
          <Esc> cancel search
    """
​
    print(helptxt)
​
    main_app()
​
​
main()
