custom_css = """
.feedback {font-size: 20px !important}
.markdown-text {
    font-size: 1em !important;
}
#models-to-add-text {
    font-size: 18px !important;
}
#citation-button span {
    font-size: 16px !important;
}
#citation-button textarea {
    font-size: 16px !important;
}
#citation-button > label > button {
    margin: 6px;
    transform: scale(1.3);
}
#leaderboard-table {
    margin-top: 15px
}
#leaderboard-table-lite {
    margin-top: 15px
}
#search-bar-table-box > div:first-child {
    background: none;
    border: none;
}

#search-bar {
    padding: 0px;
}
/* Hides the final AutoEvalColumn */
#llm-benchmark-tab-table table td:last-child,
#llm-benchmark-tab-table table th:last-child {
    display: none;
}
/* Limit the width of the first AutoEvalColumn so that names don't expand too much */
table td:first-child,
table th:first-child {
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
}
.tab-buttons button {
    font-size: 1.25em;
}
#scale-logo {
    border-style: none !important;
    box-shadow: none;
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 600px;
}
#scale-logo .download {
    display: none;
}
#filter_type{
    border: 0;
    padding-left: 0;
    padding-top: 0;
}
#filter_type label {
    display: flex;
}
#filter_type label > span{
    margin-top: var(--spacing-lg);
    margin-right: 0.5em;
}
#filter_type label > .wrap{
    width: 103px;
}
#filter_type label > .wrap .wrap-inner{  
    padding: 2px;
}
#filter_type label > .wrap .wrap-inner input{
    width: 1px
}
#filter-columns-type{
    border:0;
    padding:0.5;
}
#filter-columns-size{
    border:0;
    padding:0.5;
}
#box-filter > .form{
    border: 0
}
"""

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
    }
    """
