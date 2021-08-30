import tkinter
import os
import pickle
import functools
import sys

from PIL import Image, ImageTk


def load_imgs(d, wk=None):
    with open('vettingschedule.ls', 'r') as f:
        fltr = f.read().splitlines()

    if wk is not None:
        start = fltr.index(f'week {wk}') + 1
        end = len(fltr)
        for i in range(start, end):
            if fltr[i].startswith(f'week '):
                end = i
                break
        fltr = fltr[start:end]
    print(f'Week {wk}: {len(fltr) - 1} items')
    print(f'{fltr[1]} - {fltr[-1]}')

    all_files = os.listdir(d)
    if fltr:
        fs = [f for f in fltr if f in all_files]
    else:
        fs = [f for f in all_files if os.path.isfile(os.path.join(d, f)) and f.endswith('.png')]

    diff = len(fltr) - len(fs)
    if diff:
        print(f'{diff} missing files')
        print(set(fltr) - set(fs))

    return fs


def load():
    try:
        with open('labels.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return dict()


@functools.lru_cache(maxsize=200)
def loadim(f, w):
    im = Image.open(os.path.join('rpt', f))
    maxw = w.winfo_screenwidth()
    maxh = w.winfo_screenheight() - 260
    prop = min(1.0, maxw / im.width, maxh / im.height)
    if prop < 1.0:
        im.thumbnail((im.width * prop, im.height * prop), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image=im), im


def parseargs():
    wk_no = 0
    user_str = ''
    user_id = 0
    args = sys.argv[1:]
    if len(args) > 0:
        wk_no = args[0]
    if len(args) > 1:
        user_str = args[1]
    if len(args) > 2:
        user_id = args[2]
    return wk_no, user_str, user_id


def main_app():
    wk, usr, uid = parseargs()

    fs = load_imgs('rpt', wk)
    i = 0

    lbl = load()

    while fs[i] in lbl:
        i += 1
        if i >= len(fs):
            i = 0
            break

    searchmode = False
    search_text = ''

    def fw(skip_to_unlabeled=False):
        nonlocal i
        i += 1
        i %= len(fs)
        if skip_to_unlabeled:
            oldi = i
            while fs[i] in lbl:
                i += 1
                if i >= len(fs):
                    i = oldi
                    break
        update()

    def bk():
        nonlocal i
        i -= 1
        i %= len(fs)
        update()

    def save():
        fname = 'labels.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(lbl, f, pickle.HIGHEST_PROTOCOL)
            set_status(f'saved to {fname}')
            return f.name

    def export():
        if usr:
            fname = f'{usr}dispositions-week{wk}.csv'
        else:
            fname = 'dispositions.csv'
        with open(fname, 'wt') as f:
            for k in fs:
                if k in lbl:
                    if k.startswith('TIC'):
                        si = 3
                    else:
                        si = 0
                    f.write(f'{k[si:-4]},{uid},{lbl[k].upper()}\n')
            set_status(f'exported to {fname}')
            return f.name

    def label_str():
        if searchmode:
            sf = [f for f in fs if f.startswith(search_text)]
            n = len(sf)
            return f'Search: {search_text} ({n} matches)'

        if fs[i] in lbl:
            return f'{lbl[fs[i]].upper()}'
        return f'-'

    def status_str():
        return f'{i + 1} / {len(fs)} ({len(lbl)} labels)'

    w = tkinter.Tk()
    w.tk.call('tk', 'scaling', 2.0)

    tki, im = loadim(fs[i], w)
    pl = tkinter.Label(w, image=tki, width=im.width, height=im.height)
    pl.pack()

    txt = tkinter.Label(w, text=label_str(), width=40, height=3)
    txt.config(font=("Arial", 30))
    txt.config(fg="green")
    txt.pack()

    txts = tkinter.Label(w, text=status_str(), width=40, height=2)
    txts.config(font=("Arial", 20))
    txts.config(fg="blue")
    txts.pack()

    txtm = tkinter.Label(w, text='', width=40, height=1)
    txtm.pack()

    def set_status(msg):
        txtm.configure(text=msg)
        w.update()

    def update():
        set_status('loading')
        update_text()
        tki, _ = loadim(fs[i], w)
        pl.configure(image=tki)
        pl.image = tki
        set_status('')

    def update_text():
        txt.configure(text=label_str())
        txts.configure(text=status_str())



    def keydown(e):
        nonlocal searchmode, search_text, i
        if searchmode:
            if e.keycode == 3342463:
                search_text = search_text[:-1]
            elif e.keycode == 3473435:
                searchmode = False
            elif e.keycode == 2359309:
                sf = [f for f in fs if f.startswith(search_text)]
                i = fs.index(sf[0])
                searchmode = False
                update()
            elif e.char >= '0' and e.char <= '9':
                search_text += e.char
            update_text()
            return

        if e.char == 'q':
            save()
            w.quit()
        elif e.char == 'w':
            save()
        elif e.char == 'x':
            export()
        elif e.char in ('e', 'j', 'n', 'b', 's'):
            lbl[fs[i]] = e.char
            update_text()
            w.update()
            fw(True)

        elif e.char == '/':
            searchmode = True
            search_text = 'TIC'
            update_text()

        elif e.keycode == 3342463:
            if fs[i] in lbl:
                del lbl[fs[i]]

        elif e.keycode in (8189699, 8255233):
            fw()
        elif e.keycode in (8124162, 8320768):
            bk()

        else:
            print(e.char, e.keycode)


    w.bind("<Key>", keydown)

    w.mainloop()

    unk = []
    msg = ''


def main():
    helptxt = """
    Usage:
      python3 label.py <week no> <user initials> <user id>

    Keys:
      Arrow keys - navigate
      q - save labels and quit
      w - save labels
      x - export to CSV
      e, j, n, b, s - set label and move to next
      / - search
          <Enter> select first search match
          <Esc> cancel search
    """

    print(helptxt)

    main_app()


main()