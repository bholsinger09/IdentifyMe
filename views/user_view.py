

# ttkbootstrap UI: Black background, button at bottom center to open file dialog
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

class IdentifyMeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IdentifyMe")
        self.root.geometry("400x600")
        self.root.configure(bg="black")

        # Initialize style before using it
        style = tb.Style()
        style.configure("Black.TFrame", background="black")
        style.configure(
            "Custom.TButton",
            foreground="red",
            background="#0d6efd",
            font=("Arial", 16),
            padding=10
        )  # Bootstrap blue


        # Add a label with blue text
        self.label = tb.Label(
            self.root,
            text="Welcome to IdentifyMe!",
            bootstyle="info",
            font=("Arial", 20)
        )
        self.label.pack(pady=20)

        # Placeholder for the image label
        self.image_label = tb.Label(self.root, background="black")
        self.image_label.pack(expand=True)

        self.last_image_path = None

        bottom_frame = tb.Frame(
            self.root,
            borderwidth=0,
            style="Black.TFrame"
        )
        bottom_frame.pack(side="bottom", fill="x", pady=40)

        self.open_button = tb.Button(
            bottom_frame,
            text="Select Picture",
            command=self.open_file_dialog,
            style="Custom.TButton"
        )
        self.open_button.pack(anchor="center", pady=(0, 10))

        self.identify_button = tb.Button(
            bottom_frame,
            text="Identify Person",
            command=self.identify_person,
            bootstyle="primary",
            state="disabled"
        )
        self.identify_button.pack(anchor="center")

    def open_file_dialog(self):
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select a Picture",
            filetypes=filetypes
        )
        if filename:
            print(f"Selected file: {filename}")
            self.last_image_path = filename
            self.display_image(filename)
            self.identify_button.config(state="normal")

    def display_image(self, path):
        # Load and resize the image to fit the window
        img = Image.open(path)
        # Get current window size
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        # Reserve space for label and button
        max_w = win_w - 60 if win_w > 100 else 300
        max_h = win_h - 200 if win_h > 200 else 300
        img.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo
        self.label.config(text="Welcome to IdentifyMe!")

    def identify_person(self):
        if self.last_image_path:
            result = self.detect_face_and_gender(self.last_image_path)
            self.label.config(text=result)

    def detect_face_and_gender(self, img_path):
        try:
            result = DeepFace.analyze(img_path=img_path, actions=['gender'], enforce_detection=True)
            # DeepFace returns a list if multiple faces are detected
            if isinstance(result, list):
                if len(result) == 0:
                    return "No person detected."
                gender = result[0].get('gender', None)
            else:
                gender = result.get('gender', None)
            if gender:
                return f"Person detected: {gender}"
            else:
                return "Person detected, but gender could not be determined."
        except Exception as e:
            if 'face' in str(e).lower():
                return "No person detected."
            return f"Detection error: {e}"


if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    IdentifyMeApp(app)
    app.mainloop()
