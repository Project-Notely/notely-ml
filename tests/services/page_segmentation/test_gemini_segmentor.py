from app.services.page_segmentation.gemini_segmentor import GeminiSegmentor


def test_gemini_segmentor():
    try:
        GeminiSegmentor().execute()
    except Exception as e:
        print(e)
        assert False
