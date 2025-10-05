from app.services.page_segmentation.gemini_segmentor import GeminiSegmentor
from PIL import Image


def test_gemini_segmentor():
    try:
        GeminiSegmentor().execute(
            image=Image.open("data/notes.png"),
            query="red text",
        )
    except Exception as e:
        print(e)
        assert False
