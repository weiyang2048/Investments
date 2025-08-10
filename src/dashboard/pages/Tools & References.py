import omegaconf
import streamlit as st
import hydra


def show_references_page(dashboard_config: dict) -> None:
    """Function to show the references page with useful URLs."""
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    st.title(dashboard_config["title"])

    # Add description from YAML if available
    if "description" in dashboard_config:
        st.markdown(f"*{dashboard_config['description']}*")

    st.markdown(dashboard_config["style_string"], unsafe_allow_html=True)

    # Add sidebar links
    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <b><i>noWei.us</i></b></a>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <b><i>weiyang2048</i></b></a>",
        unsafe_allow_html=True,
    )

    # Display references from YAML file
    for section in ["Tools", "References"]:
        st.markdown(f"## {section}")
        for item_type, items in dashboard_config[section].items():
            st.markdown(f"### **{item_type}**")
            n_cols = len(items)
            cols = st.columns(n_cols)
            if type(items) == omegaconf.dictconfig.DictConfig:
                for i, (key, value) in enumerate(items.items()):
                    with cols[i]:
                        st.markdown(f"**{key}**")
                        for item in value:
                            st.markdown(
                                f"- [{item['title']}]({item['url']})", unsafe_allow_html=True
                            )
            else:
                for i, item in enumerate(items):
                    with cols[i]:
                        st.markdown(f"- [{item['title']}]({item['url']})", unsafe_allow_html=True)
            st.markdown("")


if __name__ == "__main__":
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            overrides=["+dashboard_layout=References"],
        )

    show_references_page(config["dashboard_layout"])
