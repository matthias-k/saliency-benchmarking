from datetime import datetime
import json

from jinja2 import Environment, Markup
import markdown
from staticjinja import Site

if __name__ == '__main__':
    data = json.load(open('html/data.json'))
    contexts = [
        ('index.html', data),
    ]


    filters = {}

    md = markdown.Markdown()
    filters['markdown'] = lambda text: Markup(md.convert(text))

    def format_encoded_datetime(datetime_str, format_str):
        try:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')
        return datetime_obj.strftime(format_str)
    filters['datetime'] = format_encoded_datetime

    site = Site.make_site(
        searchpath='./html/templates',
        outpath='./html',
        contexts=contexts,
	filters=filters,
        env_kwargs={
            'trim_blocks': True,
	    'lstrip_blocks': True,
        }
    )
    site.render()
