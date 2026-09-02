from src.embeddings import TfidfEmbedder
from src.llm import MockBackend
from src.rag_chain import RagPipeline, build_prompt
from src.vectorstore import QueryResult, VectorStore


def test_build_prompt_cites_chunks_and_includes_source():
    results = [
        QueryResult(id="a::0", score=0.9, document="AVTP carries CAN frames over Ethernet.", metadata={"source": "ieee1722_avtp_communication.md"}),
        QueryResult(id="b::0", score=0.5, document="Xen hosts multiple guest OSes.", metadata={"source": "xen_hypervisor_bringup.md"}),
    ]
    system, user = build_prompt("How does AVTP work?", results)

    assert "citation" in system.lower()
    assert "[1]" in user and "[2]" in user
    assert "ieee1722_avtp_communication.md" in user
    assert "AVTP carries CAN frames" in user


def test_build_prompt_handles_no_results():
    system, user = build_prompt("anything", [])
    assert "no context chunks" in user.lower()


def _corpus_pipeline() -> RagPipeline:
    docs = {
        "avtp.md": "AVTP is a Layer 2 Ethernet protocol used to carry CAN frames like accelerator pedal angle between automotive domains.",
        "xen.md": "Xen is a type-1 hypervisor used to host Linux QNX and Android guests on one automotive SoC.",
        "build.md": "IncrediBuild distributes compilation jobs across EC2 helper machines to cut build time from 24 hours to 3 hours.",
    }
    embedder = TfidfEmbedder()
    embedder.fit(list(docs.values()))
    vectors = embedder.embed(list(docs.values()))

    store = VectorStore()
    store.add(
        ids=list(docs.keys()),
        vectors=vectors,
        documents=list(docs.values()),
        metadatas=[{"source": name} for name in docs.keys()],
    )
    return RagPipeline(store, embedder, MockBackend(), top_k=1)


def test_rag_pipeline_retrieves_most_relevant_document():
    rag = _corpus_pipeline()

    result = rag.answer("How do I reduce my build time using distributed compilation?")
    assert result.sources == ["build.md"]

    result = rag.answer("How does AVTP encode accelerator pedal angle?")
    assert result.sources == ["avtp.md"]


def test_rag_pipeline_answer_includes_mock_marker():
    rag = _corpus_pipeline()
    result = rag.answer("What is Xen used for?")
    assert "[MOCK BACKEND" in result.answer
