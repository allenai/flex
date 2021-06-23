def _get_20newsgroup_classes():
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'talk.politics.mideast': 0,
        'sci.space': 1,
        'misc.forsale': 2,
        'talk.politics.misc': 3,
        'comp.graphics': 4,
        'sci.crypt': 5,
        'comp.windows.x': 6,
        'comp.os.ms-windows.misc': 7,
        'talk.politics.guns': 8,
        'talk.religion.misc': 9,
        'rec.autos': 10,
        'sci.med': 11,
        'comp.sys.mac.hardware': 12,
        'sci.electronics': 13,
        'rec.sport.hockey': 14,
        'alt.atheism': 15,
        'rec.motorcycles': 16,
        'comp.sys.ibm.pc.hardware': 17,
        'rec.sport.baseball': 18,
        'soc.religion.christian': 19,
    }

    train_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['sci', 'rec']:
            train_classes.append(key)

    val_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['comp']:
            val_classes.append(key)

    test_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
            test_classes.append(key)

    return train_classes, val_classes, test_classes


def _get_reuters_classes():
    '''
        @return list of classes associated with each split
    '''
    # 31 classes inferred using /preprocessing/reuters.py
    classes = {'trade': 28, 'grain': 12, 'ship': 25, 'gold': 11, 'acq': 0, 'tin': 27, 'ipi': 14, 'earn': 9, 'jobs': 16, 'sugar': 26, 'cpi': 7, 'money-fx': 18, 'interest': 13, 'cocoa': 3, 'coffee': 4, 'crude': 8,
               'cotton': 6, 'livestock': 17, 'money-supply': 19, 'copper': 5, 'alum': 1, 'rubber': 24, 'nat-gas': 20, 'reserves': 22, 'bop': 2, 'gnp': 10, 'iron-steel': 15, 'orange': 21, 'retail': 23, 'wpi': 30, 'veg-oil': 29}
    sorted_classes = sorted(classes, key=lambda k: classes[k])
    train_classes = sorted_classes[:15]
    val_classes = sorted_classes[15:20]
    test_classes = sorted_classes[20:31]

    return train_classes, val_classes, test_classes


def _get_huffpost_classes():
    '''
        @return list of classes associated with each split
    '''
    # randomly sorted classes
    classes = [
        'SPORTS', 'MEDIA', 'PARENTING', 'CULTURE & ARTS', 'MONEY',
        'FOOD & DRINK', 'BLACK VOICES', 'LATINO VOICES', 'TRAVEL',
        'RELIGION', 'THE WORLDPOST', 'ARTS & CULTURE', 'IMPACT', 'ARTS',
        'STYLE', 'COMEDY', 'GOOD NEWS', 'GREEN', 'WOMEN', 'FIFTY',
        'SCIENCE', 'WORLDPOST', 'WEIRD NEWS', 'CRIME', 'QUEER VOICES',
        'HEALTHY LIVING', 'TECH', 'WEDDINGS', 'EDUCATION', 'BUSINESS',
        'ENTERTAINMENT', 'TASTE', 'POLITICS', 'WORLD NEWS', 'ENVIRONMENT',
        'DIVORCE', 'PARENTS', 'COLLEGE', 'STYLE & BEAUTY', 'WELLNESS',
        'HOME & LIVING',
    ]
    train_classes = classes[:20]
    val_classes = classes[20:25]
    test_classes = classes[25:41]

    return train_classes, val_classes, test_classes


def _get_fewrel_classes():
    '''
        @return list of classes associated with each split
    '''
    # Computed using /preprocessing/fewrel.py
    bao_labels_to_wikidata_properties = {
        0: 'P931',
        1: 'P4552',
        2: 'P140',
        3: 'P1923',
        4: 'P150',
        5: 'P6',
        6: 'P27',
        7: 'P449',
        8: 'P1435',
        9: 'P175',
        10: 'P1344',
        11: 'P39',
        12: 'P527',
        13: 'P740',
        14: 'P706',
        15: 'P84',
        16: 'P495',
        17: 'P123',
        18: 'P57',
        19: 'P22',
        20: 'P178',
        21: 'P241',
        22: 'P403',
        23: 'P1411',
        24: 'P135',
        25: 'P991',
        26: 'P156',
        27: 'P176',
        28: 'P31',
        29: 'P1877',
        30: 'P102',
        31: 'P1408',
        32: 'P159',
        33: 'P3373',
        34: 'P1303',
        35: 'P17',
        36: 'P106',
        37: 'P551',
        38: 'P937',
        39: 'P355',
        40: 'P710',
        41: 'P137',
        42: 'P674',
        43: 'P466',
        44: 'P136',
        45: 'P306',
        46: 'P127',
        47: 'P400',
        48: 'P974',
        49: 'P1346',
        50: 'P460',
        51: 'P86',
        52: 'P118',
        53: 'P264',
        54: 'P750',
        55: 'P58',
        56: 'P3450',
        57: 'P105',
        58: 'P276',
        59: 'P101',
        60: 'P407',
        61: 'P1001',
        62: 'P800',
        63: 'P131',
        64: 'P177',
        65: 'P364',
        66: 'P2094',
        67: 'P361',
        68: 'P641',
        69: 'P59',
        70: 'P413',
        71: 'P206',
        72: 'P412',
        73: 'P155',
        74: 'P26',
        75: 'P410',
        76: 'P25',
        77: 'P463',
        78: 'P40',
        79: 'P921',
    }
    # head=WORK_OF_ART validation/test split
    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    def _f(lst):
        return [bao_labels_to_wikidata_properties[i] for i in lst]
    return _f(train_classes), _f(val_classes), _f(test_classes)
