{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{# ``deap_pettingzoo.ui.app`` is a Streamlit entry-point script that runs UI
   code (``st.set_page_config`` etc.) at import. It is never imported outside a
   ``streamlit run`` process, so exclude it from the API docs — importing it here
   would execute the app and flood the build log. ``modules`` holds short names,
   so qualify with ``fullname`` to avoid also dropping ``service.app``. #}
{% for item in modules if (fullname ~ '.' ~ item) != 'deap_pettingzoo.ui.app' %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
