import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.services import OpenAIService
from src.utils import setup_logger

logger = setup_logger(__name__)

def build_prompt(document_informations:str = None) -> str:
    """
    Build the system prompt specifically for reading and extracting
    structured information from images of documents.
    """
    prompt = """
    You are an assistant specialized in reading and extracting structured information 
    from images of technical and contractual documents.

    Your task:
    - Perform a precise reading of the provided document image.
    - Extract text accurately, including any **tables, labels, and structured data**.
    - Return only the extracted content.
    """

    if document_informations:
        prompt += f"\nAdditional informations about the document: {document_informations}\n"
        prompt += f"Use this to guide your final output."
    return prompt

def _process_single_image(img_b64: str, 
                          service: OpenAIService, 
                          index: int, 
                          max_tokens: int,
                          document_informations: str = None) -> str:
    """
    Process a single image by calling the model with OCR instructions.
    Includes logging for start and completion.
    """
    logger.info(f"Starting processing of image {index}")
    try:
        content_parts = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        }]

        system_prompt = build_prompt(document_informations=document_informations)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        response = service.invoke(messages,
                                  max_tokens=max_tokens)
        logger.info(f"Finished processing of image {index}")
        return response
    except Exception as e:
        logger.error(f"Error processing image {index}: {e}")
        return f"Error processing image {index}: {e}"

def process_document_images(
    document_content: List[Dict],
    service: OpenAIService,
    max_workers: int = 10,
    max_tokens: int = 1000,
    document_informations: str = None
) -> str:
    """
    Process document images using Azure OpenAI in parallel, but return a single continuous text.

    - Extracts all images from the input `document_content`.
    - Each image is processed independently with its own model call.
    - Uses ThreadPoolExecutor to parallelize multiple calls to the model.
    - Concatenates all results in order into a single continuous text.
    """

    # Filter only images
    images = [item["content"] for item in document_content if item.get("type") == "image"]

    if not images:
        logger.warning("No images provided for processing.")
        return "No images provided for processing."

    logger.info(f"Starting processing of {len(images)} images with {max_workers} workers.")

    results_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _process_single_image,
                img,
                service,
                idx,
                max_tokens,
                document_informations=document_informations
            ): idx
            for idx, img in enumerate(images, start=1)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results_dict[idx] = future.result()
            except Exception as e:
                logger.error(f"Exception occurred for image {idx}: {e}")
                results_dict[idx] = f"Exception for image {idx}: {e}"

    # Concatenate results in the original order
    continuous_text = "\n".join(results_dict[idx] for idx in sorted(results_dict.keys()))

    logger.info("Finished processing all images.")
    return continuous_text