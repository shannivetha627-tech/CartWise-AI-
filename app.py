import streamlit as st
import pandas as pd

from recommendation_engine import recommend_products

# Page configuration
st.set_page_config(
    page_title="CartWise AI",
    page_icon="🛒",
    layout="wide"
)

# Load product data
@st.cache_data
def load_products():
    return pd.read_csv("data/processed/product_price_lookup.csv")


products = load_products()

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    font-size: 38px;
    font-weight: 800;
    color: #17324d;
}
.subtitle {
    color: #64748b;
    font-size: 16px;
}
div.stButton > button {
    background-color: #176b52;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🛒 CartWise AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Smart shopping recommendations powered by machine learning and your budget.</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.header("🛍️ Shopping Preferences")
st.sidebar.write("Choose a product and set your additional budget.")

product_options = {
    f"{row['StockCode']} - {row['Description']} (Price: {row['UnitPrice']:.2f})":
    row["StockCode"]
    for _, row in products.iterrows()
}

selected_labels = st.sidebar.multiselect(
    "Select products you already have:",
    options=list(product_options.keys())
)

selected_products = [
    product_options[label] for label in selected_labels
]

budget = st.sidebar.number_input(
    "Additional budget",
    min_value=0.0,
    value=5.0,
    step=0.5
)

max_recommendations = st.sidebar.slider(
    "Maximum recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# Main content
st.subheader("Your Shopping Assistant")

col1, col2, col3 = st.columns(3)

col1.metric("Available Products", len(products))
col2.metric("Selected Products", len(selected_products))
col3.metric("Additional Budget", f"{budget:.2f}")

st.divider()

if st.button("✨ Generate Recommendations", use_container_width=True):

    if not selected_products:
        st.warning("Please select at least one product.")

    else:
        try:
            recommendations, summary = recommend_products(
                selected_products,
                budget,
                max_recommendations
            )

            st.subheader("🎯 Recommended Products")

            if recommendations.empty:
                st.info(
                    "No associated products fit your available budget."
                )

            else:
                display_df = recommendations[
                    [
                        "StockCode",
                        "Description",
                        "UnitPrice",
                        "Confidence",
                        "Lift"
                    ]
                ].copy()

                display_df["Confidence"] = (
                    display_df["Confidence"] * 100
                ).round(2).astype(str) + "%"

                display_df["Lift"] = display_df["Lift"].round(2)
                display_df["UnitPrice"] = (
                    display_df["UnitPrice"].round(2)
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader("💰 Budget Summary")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Available Budget",
                f"{summary['budget']:.2f}"
            )

            c2.metric(
                "Recommended Cost",
                f"{summary['recommended_cost']:.2f}"
            )

            c3.metric(
                "Remaining Budget",
                f"{summary['remaining_budget']:.2f}"
            )

            st.progress(
                min(
                    summary["recommended_cost"] / budget
                    if budget > 0 else 0,
                    1.0
                )
            )

            st.success(
                f"Budget validation passed: "
                f"{summary['recommended_cost']:.2f} "
                f"spent out of {budget:.2f}."
            )

            st.caption(
                "Recommendations are generated from association rules "
                "mined from the synthetic retail dataset. "
                "Prices are dataset values, not live store prices."
            )

        except Exception as error:
            st.error(f"Recommendation error: {error}")

st.divider()

st.caption("CartWise AI | Budget-Aware Shopping Basket Recommendation System")