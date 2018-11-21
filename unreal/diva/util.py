import time, re, json

def read_jsonlist(filename):
    # Check https://github.com/python/cpython/blob/master/Lib/json/decoder.py#L343
    with open(filename) as f:
        json_str = f.read()

    FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
    WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)
    WHITESPACE_STR = ' \t\n\r'
    _w=WHITESPACE.match

    json_objs = []
    json_decoder = json.JSONDecoder()

    end = 0
    while end != len(json_str):
        obj, end = json_decoder.raw_decode(json_str, idx = _w(json_str, end).end())
        json_objs.append(obj)
        end = _w(json_str, end).end()

    return json_objs
