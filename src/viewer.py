#!/usr/bin/env python

import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter import filedialog


class TkImgViewer(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Tk Image Viewer')
        self.image = None
        self.img_view = None
        self.curr_img_index = 0
        self.filenames = []

        self.img_label = StringVar()
        btn_frame = Frame(self)
        Button(btn_frame, text="Open File", command=self.open_files).pack(side=LEFT)
        Button(btn_frame, text="Prev", command=self.prev_img).pack(side=LEFT)
        Button(btn_frame, text="Next", command=self.next_img).pack(side=LEFT)
        Label(btn_frame, textvariable=self.img_label).pack(side=LEFT)
        btn_frame.pack(side=BOTTOM, fill=BOTH)

        self.img_canvas = Label(self)
        self.img_canvas.pack()
        self.pack()

    def update_ui(self):
        img_path = self.filenames[self.curr_img_index].name
        self.img_label.set(img_path.split('/')[-1])

        self.image = PIL.Image.open(img_path)
        if self.image.mode == "1":                  # bitmap image
            self.img_view = PIL.ImageTk.BitmapImage(self.image, foreground="white")
        else:                                       # photo image
            self.img_view = PIL.ImageTk.PhotoImage(self.image)

        self.img_canvas.config(image=self.img_view, bg="#000000",
                               width=self.img_view.width(), height=self.img_view.height())

    def open_files(self):
        self.filenames = filedialog.askopenfiles()
        if len(self.filenames) > 0:
            self.curr_img_index = 0
            self.image = PIL.Image.open(self.filenames[0].name)
            self.update_ui()

    def prev_img(self):
        if self.curr_img_index > 0:
            self.curr_img_index -= 1
        self.update_ui()

    def next_img(self):
        if self.curr_img_index < len(self.filenames) - 1:
            self.curr_img_index += 1
        self.update_ui()


if __name__ == "__main__":
    app = TkImgViewer()
    app.mainloop()
