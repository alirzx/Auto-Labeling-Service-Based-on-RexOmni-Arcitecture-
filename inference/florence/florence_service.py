#!/usr/bin/env python3
"""
Florence-2 Inference Service (API-ready)
Fully supports all Florence-2 tasks, normalized outputs, ready for API.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import numpy as np
import copy
import random
import io
import gc

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

class Florence2InferenceService:
    def __init__(self, model_name="microsoft/Florence-2-large", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name,local_files_only=True,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.model.eval()

    # -----------------------------
    # Core generation
    # -----------------------------
    def run_example(self, task_prompt, image: Image.Image, text_input=None):
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.torch_dtype if k=="pixel_values" else None) for k,v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                early_stopping=False
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        torch.cuda.empty_cache()
        gc.collect()

        # Ensure the parsed answer is always a dict
        if not isinstance(parsed_answer, dict):
            return {task_prompt: parsed_answer}
        return parsed_answer

    # -----------------------------
    # Drawing Utilities
    # -----------------------------
    @staticmethod
    def draw_polygons(image: Image.Image, prediction: dict, fill_mask=False):
        draw = ImageDraw.Draw(image)
        polygons_list = prediction.get('polygons', [])
        labels = prediction.get('labels', [])

        # If no polygons, fallback to bboxes
        if not polygons_list and 'bboxes' in prediction:
            polygons_list = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for x1, y1, x2, y2 in prediction['bboxes']]

        for polygons, label in zip(polygons_list, labels):
            color = random.choice(colormap)
            fill_color = random.choice(colormap) if fill_mask else None
            if isinstance(polygons[0], (list, np.ndarray)):
                # Multiple points
                for poly in polygons:
                    poly_arr = np.array(poly).reshape(-1, 2)
                    if len(poly_arr) < 3:
                        continue
                    poly_flat = poly_arr.reshape(-1).tolist()
                    draw.polygon(poly_flat, outline=color, fill=fill_color)
                    draw.text((poly_flat[0]+8, poly_flat[1]+2), label, fill=color)
            else:
                poly_arr = np.array(polygons).reshape(-1).tolist()
                draw.polygon(poly_arr, outline=color, fill=fill_color)
                draw.text((poly_arr[0]+8, poly_arr[1]+2), label, fill=color)
        return image

    @staticmethod
    def draw_ocr_bboxes(image: Image.Image, prediction: dict):
        draw = ImageDraw.Draw(image)
        bboxes = prediction.get('quad_boxes', prediction.get('bboxes', []))
        labels = prediction.get('labels', [])
        for box, label in zip(bboxes, labels):
            color = random.choice(colormap)
            poly = np.array(box).reshape(-1).tolist()
            draw.polygon(poly, width=3, outline=color)
            draw.text((poly[0]+8, poly[1]+2), label, fill=color)
        return image

    @staticmethod
    def draw_bboxes(image: Image.Image, prediction: dict):
        draw = ImageDraw.Draw(image)
        bboxes = prediction.get('bboxes', [])
        labels = prediction.get('labels', [])
        for box, label in zip(bboxes, labels):
            color = random.choice(colormap)
            if len(box) == 4:
                draw.rectangle(box, outline=color, width=3)
            draw.text((box[0]+4, box[1]+2), label, fill=color)
        return image

    @staticmethod
    def convert_to_od_format(data: dict):
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', data.get('labels', []))
        return {'bboxes': bboxes, 'labels': labels}

    @staticmethod
    def pil_to_bytes(image: Image.Image, fmt="PNG"):
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        buf.seek(0)
        return buf.getvalue()

    # -----------------------------
    # High-level task runner
    # -----------------------------
    def run_task(self, image: Image.Image, task_name: str, text_input=None, visualize=True):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        results = None
        output_image = None

        # Task categories
        caption_tasks = ['Caption', 'Detailed Caption', 'More Detailed Caption']
        grounding_tasks = ['Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding']
        seg_tasks = ['Referring Expression Segmentation', 'Region to Segmentation']
        region_tasks = ['Region to Category', 'Region to Description']
        dense_tasks = ['Dense Region Caption', 'Region Proposal']

        # Run specific task
        if task_name in caption_tasks:
            task_map = {'Caption': '<CAPTION>', 'Detailed Caption': '<DETAILED_CAPTION>', 'More Detailed Caption': '<MORE_DETAILED_CAPTION>'}
            key = task_map[task_name]
            results = self.run_example(key, image)

        elif task_name in grounding_tasks:
            base_map = {'Caption + Grounding':'<CAPTION>', 'Detailed Caption + Grounding':'<DETAILED_CAPTION>', 'More Detailed Caption + Grounding':'<MORE_DETAILED_CAPTION>'}
            base_key = base_map[task_name]
            base_results = self.run_example(base_key, image)
            caption_text = base_results.get(base_key, str(base_results))
            grounding_results = self.run_example('<CAPTION_TO_PHRASE_GROUNDING>', image, caption_text)
            results = {base_key: caption_text, '<CAPTION_TO_PHRASE_GROUNDING>': grounding_results}

        elif task_name == 'Object Detection':
            raw_results = self.run_example('<OD>', image)
            results = {'<OD>': raw_results if isinstance(raw_results, dict) else {"bboxes": [], "labels": []}}

        elif task_name == 'Open Vocabulary Detection':
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            raw_results = self.run_example(task_prompt, image, text_input)
            results = {'<OPEN_VOCABULARY_DETECTION>': self.convert_to_od_format(raw_results)}

        elif task_name in dense_tasks:
            task_map = {'Dense Region Caption':'<DENSE_REGION_CAPTION>', 'Region Proposal':'<REGION_PROPOSAL>'}
            key = task_map[task_name]
            results = self.run_example(key, image)

        elif task_name == 'Caption to Phrase Grounding':
            results = self.run_example('<CAPTION_TO_PHRASE_GROUNDING>', image, text_input)

        elif task_name in seg_tasks:
            task_map = {'Referring Expression Segmentation':'<REFERRING_EXPRESSION_SEGMENTATION>', 'Region to Segmentation':'<REGION_TO_SEGMENTATION>'}
            key = task_map[task_name]
            results = self.run_example(key, image, text_input)

        elif task_name in region_tasks:
            task_map = {'Region to Category':'<REGION_TO_CATEGORY>', 'Region to Description':'<REGION_TO_DESCRIPTION>'}
            key = task_map[task_name]
            results = self.run_example(key, image, text_input)

        elif task_name == 'OCR':
            raw_results = self.run_example('<OCR>', image)
            results = {'<OCR>': raw_results if isinstance(raw_results, dict) else {"text": raw_results}}  # Fixed: Ensure dict with text

        elif task_name == 'OCR with Region':
            results = self.run_example('<OCR_WITH_REGION>', image)

        else:
            raise ValueError(f"Unknown task: {task_name}")

        print(f"[FlorenceService] Task {task_name} results: {results}")  # Added log for debugging

        # Visualization
        # -----------------------------
        if visualize:
            output_image = copy.deepcopy(image)
            # Unwrap nested task dict for drawing
            draw_data = None
            if len(results) == 1:
                draw_data = list(results.values())[0]
            
            if task_name in seg_tasks:
                output_image = self.draw_polygons(output_image, draw_data, fill_mask=True)
            elif task_name == 'OCR with Region':
                output_image = self.draw_ocr_bboxes(output_image, draw_data)
            elif task_name in ['Object Detection', 'Open Vocabulary Detection', *region_tasks, *dense_tasks]:
                if draw_data is not None:
                    if 'bboxes' in draw_data or 'quad_boxes' in draw_data:
                        output_image = self.draw_bboxes(output_image, draw_data)
                    elif 'polygons' in draw_data:
                        output_image = self.draw_polygons(output_image, draw_data)

        # Ensure results are dict
        if not isinstance(results, dict):
            results = {task_name: results}

        # Convert image to bytes
        image_bytes = self.pil_to_bytes(output_image) if output_image is not None else None

        return {"results": results, "image_bytes": image_bytes}

    # -----------------------------
    # API-ready byte input
    # -----------------------------
    def run_task_from_bytes(self, image_bytes: bytes, task_name: str, text_input=None, visualize=True):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.run_task(image, task_name, text_input=text_input, visualize=visualize)