from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import DetrImageProcessor, DetrForSegmentation
import json
import os

def get_person_mask(image_path):
    """
    Returns a binary mask (numpy array) where 1=person, 0=background using DETR panoptic segmentation.
    """
    import torch
    from PIL import Image
    import numpy as np
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    mask = np.array(processed['segmentation'])
    # Find all segment ids for 'person' label
    person_ids = [s['id'] for s in processed['segments_info'] if s['label_id'] == 1]
    person_mask = np.isin(mask, person_ids).astype(np.uint8)
    return person_mask

import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageFilter

class IdentifyMeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IdentifyMe")
        self.root.geometry("900x1100")  # Increased window size

        # Set a dark ttkbootstrap theme for a dark UI
        style = tb.Style("superhero")  # Try "darkly", "cyborg", etc. for other dark themes
        style.configure("Black.TFrame", background="black")
        style.configure(
            "Custom.TButton",
            foreground="red",
            background="#0d6efd",
            font=("Arial", 16),
            padding=10
        )

        self.root.configure(bg="black")

        self.label = tb.Label(
            self.root,
            text="Welcome to IdentifyMe!",
            bootstyle="info",
            font=("Arial", 20),
            background="black"
        )
        self.label.pack(pady=20, fill="x")

        from tkinter import StringVar
        self.prompt_var = StringVar()
        prompt_frame = tb.Frame(self.root, style="Black.TFrame")
        prompt_frame.pack(fill="x", pady=(0, 10))
        prompt_label = tb.Label(prompt_frame, text="Prompt:", bootstyle="info", font=("Arial", 12), background="black")
        prompt_label.pack(side="left", padx=5)
        prompt_entry = tb.Entry(prompt_frame, textvariable=self.prompt_var, width=30)
        prompt_entry.pack(side="left", padx=5)

        self.redraw_clothes_button = tb.Button(
            prompt_frame,
            text="Redraw Clothes",
            command=self.redraw_clothes,
            bootstyle="warning",
            state="disabled"
        )
        self.redraw_clothes_button.pack(side="left", padx=5)

        self.correction_var = tb.StringVar()
        correction_frame = tb.Frame(self.root, style="Black.TFrame")
        correction_frame.pack(fill="x", pady=(0, 10))
        correction_label = tb.Label(
            correction_frame, text="Correction:", bootstyle="info", font=("Arial", 12), background="black"
        )
        correction_label.pack(side="left", padx=5)
        correction_entry = tb.Entry(correction_frame, textvariable=self.correction_var, width=30)
        correction_entry.pack(side="left", padx=5)

        self.save_correction_button = tb.Button(
            correction_frame,
            text="Save Correction",
            command=self.save_correction,
            bootstyle="secondary"
        )
        self.save_correction_button.pack(side="left", padx=5)

        self.gen_prompt_btn = tb.Button(
            self.root,
            text="Generate Person from Prompt",
            command=self.generate_person_from_prompt,
            bootstyle="primary"
        )
        self.gen_prompt_btn.pack(pady=5)

        self.image_label = tb.Label(self.root, background="black")
        self.image_label.pack(fill="both", expand=True)

        from tkinter import IntVar
        self.steps_var = IntVar(value=30)
        steps_frame = tb.Frame(self.root, style="Black.TFrame")
        steps_frame.pack(fill="x", pady=(0, 10))
        steps_label = tb.Label(steps_frame, text="Steps:", bootstyle="info", font=("Arial", 12), background="black")
        steps_label.pack(side="left", padx=5)
        steps_spin = tb.Spinbox(
            steps_frame,
            from_=10,
            to=100,
            increment=1,
            textvariable=self.steps_var,
            width=5
        )
        steps_spin.pack(side="left", padx=5)

    def _enforce_single_subject(self, prompt):
        # Add logic to enforce single subject in the prompt
        if "single" not in prompt and "one" not in prompt and "solo" not in prompt:
            prompt = "a single person, " + prompt
        return prompt

    def generate_person_from_prompt(self):
        """
        Generate an image from the prompt only, using Stable Diffusion text-to-image pipeline.
        """
        prompt = self.prompt_var.get()
        prompt = self._enforce_single_subject(prompt)
        if not prompt:
            import tkinter.messagebox as mb
            mb.showerror("Prompt Required", "Please enter a prompt.")
            return
        import torch
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        import os
        steps = self.steps_var.get()
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None
            )
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
            pipe = pipe.to("cpu")
            result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Generation Error", f"Stable Diffusion generation failed: {e}")
            return
        # Save generated image to data/originals/
        originals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "originals"))
        os.makedirs(originals_dir, exist_ok=True)
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', prompt)[:40] + ".png"
        save_path = os.path.join(originals_dir, safe_name)
        result.save(save_path)
        self.last_image_path = save_path
        self.last_prompt = prompt  # <-- Remember the prompt for this image
        self.display_image(result)
        self.label.config(text=f"Generated from prompt: {prompt}")

    def detect_person_hf(self):
        if not self.last_image_path:
            return
        from PIL import Image
        import numpy as np
        img = Image.open(self.last_image_path).convert("RGB")
        img_np = np.array(img)
        mask = get_person_mask(self.last_image_path)
        # Overlay mask in red
        overlay = img_np.copy()
        overlay[mask == 1] = [255, 0, 0]  # Red for person
        overlay_img = Image.fromarray(overlay)
        self._show_scrollable_image(overlay_img)
        self.label.config(text="Person detected (DETR panoptic)")
        self.clothes_button.config(state="normal")

    def _on_mousewheel(self, event):
        if hasattr(self, 'canvas'):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def segment_person_mask(self, img):
        # Use DeepLabV3+ for person segmentation
        model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        preprocess = T.Compose([
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        # In COCO, class 15 is 'person'
        mask = (output_predictions == 15)
        return mask

    def redraw_without_clothes(self):
        if not self.last_image_path:
            return
        img = Image.open(self.last_image_path).convert('RGB')
        mask = self.segment_person_mask(img)
        img_np = np.array(img)
        # Resize mask to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_resized = mask_pil.resize(img.size, resample=Image.NEAREST)
        mask_np = np.array(mask_resized) > 0
        mask_3ch = np.stack([mask_np]*3, axis=2)
        # Remove clothes (remove person region)
        no_clothes_img = np.where(mask_3ch, 0, img_np).astype(np.uint8)
        no_clothes_pil = Image.fromarray(no_clothes_img)
        self.photo = ImageTk.PhotoImage(no_clothes_pil)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo
        self.label.config(text="Redrawn without clothes.")

    def remove_clothes(self):
        if not self.last_image_path:
            return
        img = Image.open(self.last_image_path).convert('RGB')
        mask = self.segment_person_mask(img)
        img_np = np.array(img)
        # Resize mask to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_resized = mask_pil.resize(img.size, resample=Image.NEAREST)
        mask_np = np.array(mask_resized) > 0
        mask_3ch = np.stack([mask_np]*3, axis=2)
        # Remove clothes (remove person region)
        no_clothes_img = np.where(mask_3ch, 0, img_np).astype(np.uint8)
        no_clothes_pil = Image.fromarray(no_clothes_img)
        self._show_scrollable_image(no_clothes_pil)
        self.label.config(text="Clothes removed.")
        self.redraw_no_clothes_button.config(state="normal")

    def strip_clothes_and_redraw(self):
        # Placeholder: In real use, would use a generative model to reconstruct body parts
        if not self.last_image_path:
            return
        img = Image.open(self.last_image_path).convert('RGB')
        mask = self.segment_person_mask(img)
        img_np = np.array(img)
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_resized = mask_pil.resize(img.size, resample=Image.NEAREST)
        mask_np = np.array(mask_resized) > 0
        # For demo: show only the person region, fill rest with skin color
        skin_color = np.array([255, 224, 189], dtype=np.uint8)  # light skin tone
        naked_img = np.where(np.stack([mask_np]*3, axis=2), img_np, skin_color)
        naked_pil = Image.fromarray(naked_img)
        self._show_scrollable_image(naked_pil)
        self.label.config(text="Redrawn as if naked (demo placeholder).")

    def _show_scrollable_image(self, pil_img):
        # Remove previous scrollable widgets if they exist
        if hasattr(self, 'scroll_frame'):
            self.scroll_frame.destroy()
        self.scroll_frame = tb.Frame(self.root, style="Black.TFrame")
        self.scroll_frame.pack(fill="both", expand=True)
        self.canvas = tb.Canvas(self.scroll_frame, bg="black", highlightthickness=0)
        self.v_scroll = tb.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content_frame = tb.Frame(self.canvas, style="Black.TFrame")
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        # Add the image to the scrollable frame
        self.photo = ImageTk.PhotoImage(pil_img)
        img_label = tb.Label(self.content_frame, image=self.photo, background="black")
        img_label.pack(expand=True)
        # Configure scrolling region
        self.content_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def display_image(self, img_or_path):
        # Accepts either a file path or a PIL Image object
        from PIL import Image
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        # Remove previous scrollable widgets if they exist
        if hasattr(self, 'scroll_frame'):
            self.scroll_frame.destroy()
        self.scroll_frame = tb.Frame(self.root, style="Black.TFrame")
        self.scroll_frame.pack(fill="both", expand=True)
        self.canvas = tb.Canvas(self.scroll_frame, bg="black", highlightthickness=0)
        self.v_scroll = tb.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content_frame = tb.Frame(self.canvas, style="Black.TFrame")
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Increase max_w and max_h for larger images
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        max_w = win_w - 20 if win_w > 100 else 1200   # Was 60, now 20 for more width
        max_h = win_h - 60 if win_h > 200 else 1600   # Was 200, now 60 for more height
        img_copy = img.copy()
        if img_copy.width > max_w or img_copy.height > max_h:
            img_copy.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img_copy)
        img_label = tb.Label(self.content_frame, image=self.photo, background="black")
        img_label.pack(expand=True)

        # Configure scrolling region
        self.content_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.label.config(text="Welcome to IdentifyMe!")

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
            # Try to load the prompt for this image from prompts.txt
            import os
            prompts_path = os.path.join(os.path.dirname(__file__), "..", "data", "prompts.txt")
            img_filename = os.path.basename(filename)
            prompt_found = None
            if os.path.exists(prompts_path):
                with open(prompts_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2 and parts[0] == img_filename:
                            prompt_found = parts[1]
                            break
            if prompt_found:
                self.prompt_var.set(prompt_found)
                self.last_prompt = prompt_found
            else:
                self.last_prompt = self.prompt_var.get()
            self.display_image(filename)
            self.clothes_button.config(state="disabled")
            self.detect_body_button.config(state="disabled")
            self.strip_and_redraw_button.config(state="disabled")
            self.detect_person_hf_button.config(state="normal")
            self.redraw_clothes_button.config(state="normal")

    def redraw_clothes(self):
        """
        Use SAM to segment the region based on prompt, then Stable Diffusion inpainting to redraw.
        Also appends the image filename and prompt to prompts.txt if not already present.
        """
        if not self.last_image_path or not self.prompt_var.get():
            return
        prompt = self.prompt_var.get()
        import torch
        import numpy as np
        from PIL import Image
        import cv2
        from diffusers import StableDiffusionInpaintPipeline
        import os
        # Append to prompts.txt if not already present
        prompts_path = os.path.join(os.path.dirname(__file__), "..", "data", "prompts.txt")
        img_filename = os.path.basename(self.last_image_path)
        entry = f"{img_filename}\t{prompt}\n"
        # Read existing entries
        if os.path.exists(prompts_path):
            with open(prompts_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []
        if entry not in lines:
            with open(prompts_path, "a") as f:
                f.write(entry)
        # 1. Load image
        img = Image.open(self.last_image_path).convert("RGB")
        img_np = np.array(img)
        # 2. Use SAM to get mask (for demo, use the whole person mask as clothing region)
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam_ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", "sam_vit_b_01ec64.pth")
            sam_ckpt_path = os.path.abspath(sam_ckpt_path)
            sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt_path)
            predictor = SamPredictor(sam)
            predictor.set_image(img_np)
            # For demo: use center point as input, in real use GroundingDINO or user click for prompt-based
            input_point = np.array([[img_np.shape[1]//2, img_np.shape[0]//2]])
            input_label = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            mask = masks[0]
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("SAM Error", f"Could not run SAM: {e}\nMake sure you have the SAM checkpoint in the models directory.")
            return
        # 3. Prepare mask for inpainting (convert to 255 mask)
        mask_img = Image.fromarray((mask*255).astype(np.uint8))
        mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
        # Save mask to data/masks/
        masks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "masks"))
        os.makedirs(masks_dir, exist_ok=True)
        mask_save_path = os.path.join(masks_dir, img_filename)
        mask_img.save(mask_save_path)
        # 4. Run Stable Diffusion inpainting
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float32,
                safety_checker=None  # Disable NSFW filter
            )
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))  # Proper override
            pipe = pipe.to("cpu")
            result = pipe(prompt=prompt, image=img, mask_image=mask_img).images[0]
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Inpainting Error", f"Stable Diffusion inpainting failed: {e}")
            return
        # Save inpainted result to data/targets/
        targets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "targets"))
        os.makedirs(targets_dir, exist_ok=True)
        target_save_path = os.path.join(targets_dir, img_filename)
        result.save(target_save_path)
        # 5. Show result
        self.display_image(result)
        self.label.config(text=f"Redrawn with prompt: {prompt}")

    def detect_clothes(self):
        # Clothes segmentation using NVIDIA SegFormer ADE20K model
        if not self.last_image_path:
            return
        from PIL import Image
        import numpy as np
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        img = Image.open(self.last_image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        # ADE20K class indices for clothing: pants=41, skirt=44, dress=45, coat=50, hat=35, bag=39, tie=36, scarf=37, gloves=38, belt=40, boots=42, shoes=43
        clothing_classes = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50]
        img_np = np.array(img)
        overlay = img_np.copy()
        for c in clothing_classes:
            overlay[pred_mask == c] = [0, 102, 255]  # Blue for clothes
        overlay_img = Image.fromarray(overlay)
        self._show_scrollable_image(overlay_img)
        self.label.config(text="Clothes detected (SegFormer ADE20K)")

    def save_correction(self):
        """
        Save the current image filename, prompt, and correction to corrections.txt.
        Then regenerate the image using the correction as the new prompt.
        """
        import os
        import tkinter.messagebox as mb
        if not self.last_image_path or not self.prompt_var.get() or not self.correction_var.get():
            mb.showerror("Missing Data", "Please select an image, enter a prompt, and provide a correction.")
            return
        corrections_path = os.path.join(os.path.dirname(__file__), "..", "data", "corrections.txt")
        img_filename = os.path.basename(self.last_image_path)
        prompt = self.last_prompt or self.prompt_var.get()  # <-- Use remembered prompt if available
        correction = self.correction_var.get()
        entry = f"{img_filename}\t{prompt}\t{correction}\n"
        with open(corrections_path, "a") as f:
            f.write(entry)
        mb.showinfo("Saved", "Correction saved successfully.")
        self.correction_var.set("")

        # Regenerate the image using the correction as the new prompt
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            from PIL import Image
            import re
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None
            )
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
            pipe = pipe.to("cpu")
            steps = self.steps_var.get()
            full_prompt = f"{prompt}. {correction}"  # <-- Always combine original prompt and correction
            full_prompt = self._enforce_single_subject(full_prompt)
            result = pipe(prompt=full_prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
            # Save regenerated image to data/corrections/
            corrections_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "corrections"))
            os.makedirs(corrections_dir, exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', correction)[:40] + ".png"
            save_path = os.path.join(corrections_dir, safe_name)
            result.save(save_path)
            self.display_image(result)
            self.label.config(text=f"Regenerated with correction: {correction}")
            self.last_image_path = save_path  # Update last image path
        except Exception as e:
            mb.showerror("Generation Error", f"Regeneration with correction failed: {e}")

    def save_session(self):
        session = {
            "prompt": self.prompt_var.get(),
            "correction": self.correction_var.get(),
            "steps": self.steps_var.get(),
            "last_image_path": getattr(self, "last_image_path", None),
            "last_prompt": getattr(self, "last_prompt", None)
        }
        session_path = os.path.join(os.path.dirname(__file__), "..", "data", "session.json")
        with open(session_path, "w") as f:
            json.dump(session, f)

    def load_session(self):
        session_path = os.path.join(os.path.dirname(__file__), "..", "data", "session.json")
        if os.path.exists(session_path):
            with open(session_path, "r") as f:
                session = json.load(f)
            self.prompt_var.set(session.get("prompt", ""))
            self.correction_var.set(session.get("correction", ""))
            self.steps_var.set(session.get("steps", 30))
            self.last_image_path = session.get("last_image_path", None)
            self.last_prompt = session.get("last_prompt", None)
            if self.last_image_path and os.path.exists(self.last_image_path):
                self.display_image(self.last_image_path)

if __name__ == "__main__":
    import tkinter as tk
    import os
    root = tk.Tk()
    app = IdentifyMeApp(root)
    app.load_session()  # <-- Load session on startup
    def on_closing():
        app.save_session()  # <-- Save session on exit
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
