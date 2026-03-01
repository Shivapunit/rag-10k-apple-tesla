"""
Test Script - Run all 13 evaluation questions
Outputs results to test_results.json
"""

import json
import logging
from pathlib import Path
from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 13 Test Questions from Assignment
TEST_QUESTIONS = [
    {
        "question_id": 1,
        "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?",
        "expected": "$391,036 million",
        "source": "Apple 10-K, Item 8, p. 282"
    },
    {
        "question_id": 2,
        "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?",
        "expected": "15,115,823,000 shares",
        "source": "Apple 10-K, first paragraph"
    },
    {
        "question_id": 3,
        "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?",
        "expected": "$96,662 million",
        "source": "Apple 10-K, Item 8, Note 9, p. 394"
    },
    {
        "question_id": 4,
        "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?",
        "expected": "November 1, 2024",
        "source": "Apple 10-K, Signature page"
    },
    {
        "question_id": 5,
        "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?",
        "expected": "No",
        "source": "Apple 10-K, Item 1B, p. 176"
    },
    {
        "question_id": 6,
        "question": "What was Tesla's total revenue for the year ended December 31, 2023?",
        "expected": "$96,773 million",
        "source": "Tesla 10-K, Item 7"
    },
    {
        "question_id": 7,
        "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?",
        "expected": "~84% ($81,924M / $96,773M)",
        "source": "Tesla 10-K, Item 7"
    },
    {
        "question_id": 8,
        "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?",
        "expected": "Central to strategy, innovation, leadership",
        "source": "Tesla 10-K, Item 1A"
    },
    {
        "question_id": 9,
        "question": "What types of vehicles does Tesla currently produce and deliver?",
        "expected": "Model S, Model 3, Model X, Model Y, Cybertruck",
        "source": "Tesla 10-K, Item 1"
    },
    {
        "question_id": 10,
        "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?",
        "expected": "Finance solar systems with investors; customers sign PPAs",
        "source": "Tesla 10-K, Item 7"
    },
    {
        "question_id": 11,
        "question": "What is Tesla's stock price forecast for 2025?",
        "expected": "Not answerable (out-of-scope)",
        "source": "N/A"
    },
    {
        "question_id": 12,
        "question": "Who is the CFO of Apple as of 2025?",
        "expected": "Not answerable (out-of-scope)",
        "source": "N/A"
    },
    {
        "question_id": 13,
        "question": "What color is Tesla's headquarters painted?",
        "expected": "Not answerable (out-of-scope)",
        "source": "N/A"
    },
]


def run_tests():
    """Run all test questions and save results"""

    print("\n" + "=" * 80)
    print("RAG SYSTEM - 13 QUESTION TEST SUITE")
    print("=" * 80 + "\n")

    # Initialize RAG pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(
        data_dir="data",
        vector_store_dir="vector_store",
        top_k=5,
        chunk_size=500,
        chunk_overlap=100
    )

    # Build or load index
    if not rag.is_indexed():
        print("\nBuilding vector index from PDFs...")
        print("(This may take 2-5 minutes on first run)\n")
        if not rag.build_index():
            print("❌ Failed to build index. Check that PDFs are in data/ directory.")
            return
    else:
        print("Loading existing vector index...\n")
        rag.load_index()

    print(f"✅ Index loaded: {rag.get_chunk_count()} chunks\n")

    # Run tests
    results = []

    print("Running 13 test questions...\n")
    print("=" * 80)

    for i, test_q in enumerate(TEST_QUESTIONS, 1):
        q_id = test_q["question_id"]
        question = test_q["question"]
        expected = test_q["expected"]

        print(f"\n[{i}/13] Question {q_id}")
        print(f"Q: {question[:80]}...")
        print(f"Expected: {expected}")

        try:
            result = rag.answer_question(question, top_k=5)

            answer = result["answer"]
            sources = result["sources"]

            print(f"A: {answer[:100]}...")
            print(f"Sources: {len(sources)} document(s)")

            results.append({
                "question_id": q_id,
                "question": question,
                "answer": answer,
                "expected": expected,
                "sources": [
                    {
                        "document": s["document"],
                        "item": s["item"],
                        "page": s["page"]
                    }
                    for s in sources
                ],
                "num_sources": len(sources)
            })

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "question_id": q_id,
                "question": question,
                "answer": f"Error: {str(e)}",
                "expected": expected,
                "sources": [],
                "num_sources": 0
            })

    print("\n" + "=" * 80)
    print("\n✅ ALL TESTS COMPLETED\n")

    # Summary statistics
    answered_with_sources = sum(1 for r in results if r["num_sources"] > 0)
    out_of_scope = sum(1 for r in results if r["question_id"] >= 11)

    print(f"📊 Statistics:")
    print(f"  Total Questions: {len(results)}")
    print(f"  Answered with Sources: {answered_with_sources}")
    print(f"  Out-of-Scope Questions: {out_of_scope}")

    # Save results
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}\n")

    # Print compact JSON results
    print("=" * 80)
    print("JSON OUTPUT:\n")

    compact_results = [
        {
            "question_id": r["question_id"],
            "answer": r["answer"],
            "sources": [f"{s['document']} - {s['item']} - p. {s['page']}" for s in r["sources"]]
        }
        for r in results
    ]

    print(json.dumps(compact_results, indent=2))

    return results


if __name__ == "__main__":
    run_tests()

