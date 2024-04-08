import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler")
        self.input_dir = ''
        self.output_dir = ''
        self.images = []
        self.image_count = 0
        self.current_index = 0
        self.gender = ''
        self.age = ''
        self.race = ''
        self.gender_buttons = []
        self.age_buttons = []
        self.race_buttons = []
        self.total_images = StringVar()
        self.total_images.set("Total Images Left: 0")
        self.processed_images = StringVar()
        self.processed_images.set("Processed Images: 0")
        self.create_widgets()

    def create_widgets(self):
        self.create_directory_widgets()
        self.create_image_label()
        self.create_option_buttons()
        self.create_next_button()
        self.create_counter_labels()

    def create_directory_widgets(self):
        Label(self.root, text="Input Directory:").grid(row=0, column=0)
        self.entry_input = Entry(self.root)
        self.entry_input.grid(row=0, column=1)
        Button(self.root, text="Browse", command=self.browse_input_directory).grid(row=0, column=2)

        Label(self.root, text="Output Directory:").grid(row=1, column=0)
        self.entry_output = Entry(self.root)
        self.entry_output.grid(row=1, column=1)
        Button(self.root, text="Browse", command=self.browse_output_directory).grid(row=1, column=2)

    def create_image_label(self):
        self.label_image = Label(self.root)
        self.label_image.grid(row=2, column=0, columnspan=3)

    def create_option_buttons(self):
        self.create_buttons("Gender:", ['Male', 'Female', 'Other_Gender'], 3, 0, self.gender_buttons)
        self.create_buttons("Age:", ['Young', 'Adult', 'Old'], 3, 1, self.age_buttons)
        self.create_buttons("Race:", ['White', 'Asian', 'Black', 'Hispanic', 'Other_Race'], 3, 2, self.race_buttons)

    def create_buttons(self, label_text, options, row, column, buttons):
        Label(self.root, text=label_text).grid(row=row, column=column, sticky=W)
        for i, option in enumerate(options):
            var = IntVar()
            button = Checkbutton(self.root, text=option, variable=var, command=lambda opt=option, var=var: self.select_option(opt, var), offvalue=0, onvalue=1)
            button.grid(row=row+i+1, column=column, sticky=W+E)
            buttons.append((button, var))

    def create_next_button(self):
        self.button_next = Button(self.root, text="Next", command=self.next_image)
        self.button_next.grid(row=8, column=1)
        
    def create_counter_labels(self):
        Label(self.root, textvariable=self.total_images).grid(row=9, column=0)
        Label(self.root, textvariable=self.processed_images).grid(row=9, column=1)

    def browse_input_directory(self):
        self.input_dir = filedialog.askdirectory()
        self.entry_input.delete(0, END)
        self.entry_input.insert(0, self.input_dir)
        self.load_images()
        
    def browse_output_directory(self):
        self.output_dir = filedialog.askdirectory()
        self.entry_output.delete(0, END)
        self.entry_output.insert(0, self.output_dir)
        
    def load_images(self):
        if self.input_dir:
            self.images = os.listdir(self.input_dir)
            self.image_count = len(self.images)
            self.total_images.set(f"Total Images: {self.image_count}")
            self.load_image()
        
    def load_image(self):
        image_path = os.path.join(self.input_dir, self.images[self.current_index])
        img = Image.open(image_path)
        resized_img = img.resize((192, 192))
        self.imgtk = ImageTk.PhotoImage(resized_img)
        self.label_image.configure(image=self.imgtk)
        
    def select_option(self, option, var):
        if option in ['Male', 'Female', 'Other_Gender']:
            self.gender = option if var.get() else ''
            self.update_buttons_state(self.gender_buttons, option)
        elif option in ['Young', 'Adult', 'Old']:
            self.age = option if var.get() else ''
            self.update_buttons_state(self.age_buttons, option)
        elif option in ['White', 'Asian', 'Black', 'Hispanic', 'Other_Race']:
            self.race = option if var.get() else ''
            self.update_buttons_state(self.race_buttons, option)
        
    def update_buttons_state(self, buttons, selected_option):
        for button, var in buttons:
            if button.cget("text") == selected_option and var.get():
                button.config(selectcolor="lightblue")
            else:
                button.deselect()
                var.set(0)
        
    def next_image(self):
        if self.gender and self.age and self.race:
            # renaming the image by placing tags at the end of name and writing moving the file to output directory (file will be moved not copied)
            old_name, extension = self.images[self.current_index].split('.')
            new_name = f"{old_name}_{self.gender.casefold()}_{self.age.casefold()}_{self.race.casefold()}.{extension}"
            os.rename(os.path.join(self.input_dir, old_name + '.' + extension), os.path.join(self.output_dir, new_name))
            self.current_index += 1
            self.image_count -= 1

        # updating and displaying counts
        self.total_images.set(f"Total Images Left: {self.image_count}")
        self.processed_images.set(f"Processed Images: {self.current_index}")

        #loading new img
        if self.current_index < len(self.images):
            self.load_image()
            self.label_image.configure(image=self.imgtk)
            
        # resetting the slections
        self.gender = ''
        self.age = ''
        self.race = ''
        self.update_buttons_state(self.gender_buttons, None)
        self.update_buttons_state(self.age_buttons, None)
        self.update_buttons_state(self.race_buttons, None)

root = Tk()
app = ImageLabeler(root)
root.mainloop()
