import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

root.filename = tk.filedialog.askopenfilename(initialdir="/", title="Select image",
                                              filetypes=[("Image", ".jpg .gif .png")])
print(root.filename)
