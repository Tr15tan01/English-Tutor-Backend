from ddgs import DDGS

def test_ddgs(query: str):
    print(f"\n🔍 Searching DDGS for: '{query}'\n")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                print("❌ No results.")
                return
            for i, res in enumerate(results, 1):
                print(f"{i}. {res.get('body', 'No snippet')}")
                print(f"   URL: {res.get('href', '')}")
                print("-" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ddgs("current population of Tokyo")