import os
import altair as alt
from myst_nb import glue


print('hey')
# Use PNG images in the PDF version of the books to make sure that they render
if 'BOOK_BUILD_TYPE' in os.environ and os.environ['BOOK_BUILD_TYPE'] == 'PDF':
    alt.data_transformers.disable_max_rows()
    alt.renderers.enable('png', scale_factor=0.7, ppi=300)
else:
    # Reduce chart sizes and allow to plot up to 100k graphical objects (not the same as rows in the data frame)
    print(alt.__version__)
    alt.data_transformers.enable('vegafusion')
