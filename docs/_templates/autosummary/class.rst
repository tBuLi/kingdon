{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:

   {% set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
   {% if own_methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
