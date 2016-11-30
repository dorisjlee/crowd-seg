ACCESS_KEY = 'AKIAIZQ3EQ4YZMDKA4RA'
SECRET_KEY = '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'
DEV_ENVIROMENT_BOOLEAN = False
#This allows us to specify whether we are pushing to the sandbox or live site.
if DEV_ENVIROMENT_BOOLEAN:
    AMAZON_HOST = 'mechanicalturk.sandbox.amazonaws.com'
    MastersQualID = '2F1KVCNHMVHV8E9PBUB2A4J79LU20F'
else:
    AMAZON_HOST = 'mechanicalturk.amazonaws.com'
    MastersQualID = '2NDP2L92HECWY8NS8H3CK0CP5L9GHO'
