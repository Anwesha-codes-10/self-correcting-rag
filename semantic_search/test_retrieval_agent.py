from retrieval_agent import RetrievalAgent


def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    except Exception as e:
        raise AssertionError(f"Expected {exc_type}, got {type(e)}: {e}")

    raise AssertionError(f"Expected {exc_type} to be raised")


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def encode(self, query, convert_to_numpy=True):
        return FakeEmbedding([0.1, 0.2, 0.3])


class FakeCollection:
    def __init__(self, return_empty=False):
        self.return_empty = return_empty

    def query(self, query_embeddings, n_results):
        if self.return_empty:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        documents = [
            "FastAPI routes",
            "Dependency Injection",
            "Random Forest",
        ]

        ids = ["PAGE_1", "PAGE_2", "PAGE_3"]

        distances = [0.2, 0.5, 0.8]

        return {
            "documents": [documents[:n_results]],
            "ids": [ids[:n_results]],
            "distances": [distances[:n_results]],
        }


def test_distance_to_confidence():
    agent = RetrievalAgent(FakeCollection(), FakeModel())

    assert agent._convert_distance_to_confidence(0.0) == 1.0
    assert agent._convert_distance_to_confidence(0.5) == 0.5
    assert agent._convert_distance_to_confidence(1.0) == 0.0


def test_execute_returns_dict():
    agent = RetrievalAgent(FakeCollection(), FakeModel())

    result = agent.execute("FastAPI routes")

    assert isinstance(result, dict)


def test_execute_returns_top_k_documents():
    agent = RetrievalAgent(FakeCollection(), FakeModel(), top_k=2)

    result = agent.execute("FastAPI routes")

    assert result["num_results"] <= agent.top_k


def test_document_structure():
    agent = RetrievalAgent(FakeCollection(), FakeModel())

    result = agent.execute("FastAPI")

    for doc in result["retrieved_documents"]:
        assert "content" in doc
        assert "source" in doc
        assert "distance" in doc
        assert "confidence" in doc


def test_validate_output():
    agent = RetrievalAgent(FakeCollection(), FakeModel())

    result = agent.execute("FastAPI")

    assert agent.validate_output(result) is True


def test_invalid_query():
    agent = RetrievalAgent(FakeCollection(), FakeModel())

    assert_raises(ValueError, lambda: agent.execute(""))
    assert_raises(ValueError, lambda: agent.execute("   "))


def test_no_results():
    agent = RetrievalAgent(FakeCollection(return_empty=True), FakeModel())

    result = agent.execute("unknown topic")

    assert result["num_results"] == 0
    assert result["retrieved_documents"] == []


def run_all_tests():
    print("\n" + "=" * 60)
    print("UNIT TESTS : RETRIEVAL AGENT")
    print("" + "=" * 60 + "\n")

    tests = [
        test_distance_to_confidence,
        test_execute_returns_dict,
        test_execute_returns_top_k_documents,
        test_document_structure,
        test_validate_output,
        test_invalid_query,
        test_no_results,
    ]

    failed = []

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 60)

    if not failed:
        print(f"✅ ALL TESTS PASSED ({len(tests)}/{len(tests)})")
    else:
        print(f"❌ FAILED: {failed}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
