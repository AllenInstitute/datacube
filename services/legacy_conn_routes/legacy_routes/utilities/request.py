import ast


def decode(inpt):
    
    if isinstance(inpt, bytes):
        return decode(inpt.decode())

    if isinstance(inpt, str):
        return parse_string(inpt)

    if isinstance(inpt, list) and len(inpt) != 1:
        return [ decode(sub_input) for sub_input in inpt ]

    if isinstance(inpt, list) and len(inpt) == 1:
        return decode(inpt[0]) # for some reason key=val,val1,val2 => {'key': ['val,val1,val2']}

    if isinstance(inpt, dict):
        return { decode(key): decode(value) for key, value in inpt.items() }

    raise ValueError('cannot decode: {}'.format(type(inpt)))


def parse_string(inpt):
    ''' The strings that come back from klein requests are a bit wonky. This 
    function unwonks them.
    '''

    if ',' in inpt:
        return [ parse_string(sub_inpt) for sub_inpt in inpt.split(',') ]
    
    if inpt in ('true', 'false'):
        return parse_string( inpt.capitalize() )

    try:
        return ast.literal_eval(inpt)
    except (ValueError, SyntaxError) as err:
        return inpt