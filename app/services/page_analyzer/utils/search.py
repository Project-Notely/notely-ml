from app.services.page_analyzer.models.models import TextBox


def search_text_in_image(text_boxes: list[TextBox], search_term: str) -> list[TextBox]:
    """Search for a term in the detected text boxes (ctrl-f functionality)"""
    search_term = search_term.lower()
    matching_boxes = []

    for text_box in text_boxes:
        if search_term in text_box.text.lower():
            matching_boxes.append(text_box)

    return matching_boxes
