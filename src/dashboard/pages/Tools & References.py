import omegaconf
import streamlit as st
import hydra


def show_references_page(dashboard_config: dict) -> None:
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    st.title(dashboard_config["title"])
    # % style string
    style_css_url = dashboard_config["style_css_url"]
    with open(style_css_url, "r") as f:
        style_string = f.read()
    st.markdown(f"<style>{style_string}</style>", unsafe_allow_html=True)
    # st.markdown(dashboard_config["style_string"], unsafe_allow_html=True)

    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <b><i>noWei.us</i></b></a>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <b><i>weiyang2048</i></b></a>",
        unsafe_allow_html=True,
    )

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
                            color = dashboard_config["color_dict"].get(item["title"].split(" ")[0], "rgba(100,100,255,1)")
                            st.markdown(
                                f"- <a href='{item['url']}' target='_blank' style='color: {color}'>{item['title']}</a>",
                                unsafe_allow_html=True,
                            )
            else:
                for i, item in enumerate(items):
                    with cols[i]:
                        st.markdown(
                            f"- <a href='{item['url']}' target='_blank' style='color: {color}'>{item['title']}</a>",
                            unsafe_allow_html=True,
                        )
            st.markdown("")


if __name__ == "__main__":
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            overrides=["+style_conf=References"],
        )

    show_references_page(config["style_conf"])
