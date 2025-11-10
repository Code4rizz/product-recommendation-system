import streamlit as st
import pandas as pd
from recommendation import RecommendationEngine

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="🛒 Smart Product Recommender", page_icon="🛒", layout="wide")

# ---------------------------
# Engine (cached)
# ---------------------------
@st.cache_resource
def load_engine():
    return RecommendationEngine()

engine = load_engine()

# ---------------------------
# Session State Defaults
# ---------------------------
st.session_state.setdefault("cart", [])
st.session_state.setdefault("purchase_history", [])
st.session_state.setdefault("last_results", None)
st.session_state.setdefault("last_search", "")
st.session_state.setdefault("last_price_filter", None)
st.session_state.setdefault("category_filter", "All")
st.session_state.setdefault("deep_search", False)

# ---------------------------
# Header / Hero
# ---------------------------
left, right = st.columns([4, 1], vertical_alignment="center")
with left:
    st.title("🛒 Smart Product Recommender")
    st.caption("Search products, personalize by your history, and filter by category.")
with right:
    st.metric("Cart Items", len(st.session_state.cart))

st.divider()

# ---------------------------
# Sidebar — Profile
# ---------------------------
with st.sidebar:
    st.header("👤 Your Profile")

    if st.session_state.purchase_history:
        profile = engine.get_user_profile(st.session_state.purchase_history)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Purchases", profile["total_purchases"])
        with c2:
            st.metric("Avg Spend", f"₹{profile['avg_price']:.0f}")

        if profile["categories"]:
            st.subheader("Top Categories")
            top3 = sorted(profile["categories"].items(), key=lambda x: x[1], reverse=True)[:3]
            for cat, cnt in top3:
                st.write(f"• {cat}: {cnt}")

        def _clear_history():
            st.session_state["purchase_history"] = []
            st.session_state["cart"] = []
            st.session_state["last_results"] = None
            st.session_state["last_search"] = ""
            st.session_state["last_price_filter"] = None

        st.button("Clear History", on_click=_clear_history)
    else:
        st.info("No purchases yet — your recommendations will personalize as you buy.")

# ---------------------------
# Tabs
# ---------------------------
tab_search, tab_cart = st.tabs([
    "🔎 Search & Recommend",
    f"🧺 Cart ({len(st.session_state.cart)})",
])

# ---------------------------
# Search Tab
# ---------------------------
with tab_search:
    # Search bar & filters row
    r1c1, r1c2, r1c3, r1c4 = st.columns([3, 2, 2, 1])
    with r1c1:
        search = st.text_input("Search for products", placeholder="e.g., chips under 200, rice below 500")
    with r1c2:
        categories = engine.get_categories()
        current_index = categories.index(st.session_state.category_filter) if st.session_state.category_filter in categories else 0
        st.session_state.category_filter = st.selectbox("Category", options=categories, index=current_index)
    with r1c3:
        num_results = st.number_input("Max results", min_value=3, max_value=50, value=10, step=1)
    with r1c4:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    # Search options row
    col1, col2 = st.columns([1, 1])
    with col1:
        personalize = st.toggle(
            "🌟 Personalize using my purchase history",
            value=bool(st.session_state.purchase_history),
            help="Boosts products from categories you've purchased before (price-aware).",
        )
    with col2:
        st.session_state.deep_search = st.toggle(
            "🚀 Deep Search (FAISS)",
            value=st.session_state.deep_search,
            help="Uses FAISS for 10-30x faster approximate search. Best for large catalogs or quick browsing.",
        )

    # Price filter examples
    with st.expander("💡 Price Filter Examples"):
        st.markdown("""
        Try these natural language price filters:
        - `chips under 200` or `chips below 200`
        - `rice above 500` or `rice over 500`
        - `snacks between 50 and 150`
        - `book less than 300`
        - `oil more than 100`
        """)

    # On search — compute & persist results
    if search and search_btn:
        user_history = None
        if personalize and st.session_state.purchase_history:
            profile = engine.get_user_profile(st.session_state.purchase_history)
            user_history = {
                "categories": list(profile["categories"].keys()),
                "subcategories": list(profile["subcategories"].keys()),
                "avg_price": profile["avg_price"],
            }

        search_mode = "🚀 FAISS" if st.session_state.deep_search else "🔍 Exact"
        with st.spinner(f"Searching with {search_mode} mode..."):
            results, price_filter, err = engine.get_recommendations(
                user_query=search,
                top_n=int(num_results),
                category_filter=st.session_state.category_filter,
                min_similarity=0,
                user_purchase_history=user_history,
                deep_search=st.session_state.deep_search,
            )

        if err:
            st.error(f"❌ Search failed: {err}")
        else:
            st.session_state.last_results = results
            st.session_state.last_search = search
            st.session_state.last_price_filter = price_filter

    # Always show last results if available
    results = st.session_state.last_results
    if results is None:
        st.info("💡 Tip: Toggle **Deep Search** for 10-30x faster results on large catalogs!")
    else:
        search_badge = "🚀 FAISS" if st.session_state.deep_search else "🔍 Exact"
        st.write(
            f"### Results for: *{st.session_state.last_search}* {search_badge}  "
            f"(Category: {st.session_state.category_filter})"
        )
        
        # Show active price filter
        if st.session_state.last_price_filter:
            filter_text = []
            if 'min' in st.session_state.last_price_filter:
                filter_text.append(f"≥₹{st.session_state.last_price_filter['min']}")
            if 'max' in st.session_state.last_price_filter:
                filter_text.append(f"≤₹{st.session_state.last_price_filter['max']}")
            st.info(f"💰 Price filter active: {' & '.join(filter_text)}")

        if len(results) == 0:
            st.warning("⚠️ No products found with current filters. Try a different search or category.")
        else:
            for i, row in results.reset_index(drop=True).iterrows():
                box = st.container(border=True)
                with box:
                    c1, c2, c3 = st.columns([6, 2, 2])

                    with c1:
                        # Star for personalized results
                        star = "⭐ " if bool(row.get("is_personalized", False)) else ""
                        st.markdown(f"**{star}{row['Name']}**")
                        st.caption(f"{row['Category']} › {row['SubCategory']}")
                        
                        # Show personalized score (capped display at 100%)
                        try:
                            sim_val = float(row.get("personalized_score", row.get("similarity_score", 0)))
                            display_val = min(sim_val, 100.0)
                        except Exception:
                            sim_val = 0.0
                            display_val = 0.0
                        
                        progress_value = min(max(sim_val / 100.0, 0.0), 1.0)
                        st.progress(progress_value, text=f"Match Score: {display_val:.1f}%")

                    with c2:
                        st.metric("Price", f"₹{float(row['Price']):.0f}")

                    with c3:
                        btn_key = f"add_{i}_{row['Name']}"
                        if st.button("Add to cart", key=btn_key, use_container_width=True):
                            st.session_state.cart.append(
                                {
                                    "name": row["Name"],
                                    "price": float(row["Price"]),
                                    "category": row["Category"],
                                    "subcategory": row["SubCategory"],
                                }
                            )
                            st.toast("✓ Added to cart", icon="🧺")
                            st.rerun()

            # Show legend
            legend_items = []
            if results['is_personalized'].any():
                legend_items.append("⭐ = Personalized")
            legend_items.append(f"🚀 = FAISS" if st.session_state.deep_search else "🔍 = Exact Search")
            st.caption(" | ".join(legend_items))

# ---------------------------
# Cart Tab
# ---------------------------
with tab_cart:
    if not st.session_state.cart:
        st.info("🛒 Your cart is empty. Add some products from the search tab!")
    else:
        total = 0.0
        for i, item in enumerate(st.session_state.cart):
            row = st.container(border=True)
            with row:
                c1, c2, c3 = st.columns([6, 2, 1])
                with c1:
                    st.markdown(f"**{item['name']}**")
                    st.caption(f"{item['category']} › {item['subcategory']}")
                with c2:
                    st.write(f"₹{float(item['price']):.0f}")
                with c3:
                    if st.button("✕", key=f"rm_{i}", help="Remove from cart"):
                        st.session_state.cart.pop(i)
                        st.rerun()
            total += float(item["price"])

        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Total", f"₹{total:.2f}")
        with col2:
            if st.button("🛍️ Purchase", type="primary", use_container_width=True):
                st.session_state.purchase_history.extend(st.session_state.cart)
                st.session_state.cart = []
                st.success("✅ Purchase completed! Your recommendations will now be personalized.")
                st.balloons()
                st.rerun()