import streamlit as st
from typing import Callable


def setup_page_and_sidebar(dashboard_config: dict, add_to_sidebar: Callable = None) -> None:
    """
    Set up the Streamlit page configuration, including title, description,
    custom CSS styling, and sidebar links.

    Parameters
    ----------
    dashboard_config : dict
        A dictionary containing dashboard configuration options.
        Must include a "page_config" key for Streamlit's set_page_config,
        and may include "title", "description", and "style_css_url".
    add_to_sidebar : Callable, optional
        A function to add custom sidebar elements. If provided, it will be called
        and its return value will be returned by this function.

    Returns
    -------
    Any
        The return value of add_to_sidebar(), if provided; otherwise, None.

    Example
    -------
    >>> setup_page_and_sidebar(config["style_conf"], add_to_sidebar=my_sidebar_func)
    """
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    title = dashboard_config.get("title", None)
    if title:
        st.title(title)
    description = dashboard_config.get("description", None)
    if description:
        st.markdown(description)

    # Load and apply custom CSS styling from the provided file path
    style_css_url = dashboard_config["style_css_url"]
    with open(style_css_url, "r") as f:
        style_string = f.read()
    st.markdown(f"<style>{style_string}</style>", unsafe_allow_html=True)

    res = None
    if add_to_sidebar:
        res = add_to_sidebar()

    # Add homepage and LinkedIn links to the sidebar
    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <b><i>noWei.us</i></b></a>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <b><i>weiyang2048</i></b></a>",
        unsafe_allow_html=True,
    )

    return res
