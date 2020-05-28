import base64, zlib, json


def decode(blueprint):
    assert blueprint and blueprint != "", "is null or empty"
    t = blueprint[1:]
    t = base64.b64decode(t)
    t = zlib.decompress(t)
    t = json.loads(t)
    return t


def encode(blueprint_dict):
    t = json.dumps(blueprint_dict)
    t = t.encode()
    t = zlib.compress(t)
    t = base64.b64encode(t)
    t = b"0" + t
    t = t.decode()
    return t


if __name__ == "__main__":
    import sys, pprint

    bp = sys.argv[1]
    pprint.pprint(decode(bp))
