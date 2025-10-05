import json
import os

from app.services.query_parser.query_parser import QueryParser


async def test_query_parser():
    try:
        parser = QueryParser()
        query = await parser.execute(
            "i want you to find the main title and the first paragraph in the document and other similar stuff"
        )
        print(query)

        # Define the output directory and ensure it exists
        output_dir = "tests/results"
        os.makedirs(output_dir, exist_ok=True)

        # Save the query result to a file
        with open(os.path.join(output_dir, "query_parser_result.json"), "w") as f:
            json.dump(query.model_dump(), f, indent=4)

        assert query != ""
    except Exception as e:
        print(e)
        assert False
