import re
import pickle
from config import *

# Load ordered ICD-10 sequences per patient produced in previous steps
with open(f"../{DATASET_DIR}/icd10_order_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}.pkl", "rb") as f:
    date_dict = pickle.load(f)

# Helpers to map/normalize ICD-10 codes
def get_icd10_category(icd10_code,level=None):
    icd10_code = icd10_code.strip()
    patterns = {
        3: r'([A-Z]\d{2})',
        4: r'([A-Z]\d{2}\.[0-9X])',
    }
    pattern = patterns[level]
    match = re.match(pattern, icd10_code, re.IGNORECASE)
    return match.group(1).upper() if match else icd10_code.upper()
def map_icd10_to_chapter(icd10_code,level=None):
    code = icd10_code.strip().upper()
    chapters = [
        ('A00', 'B99', 1),
        ('C00', 'D48', 2),
        ('D50', 'D89', 3),
        ('E00', 'E90', 4),
        ('F00', 'F99', 5),
        ('G00', 'G99', 6),
        ('H00', 'H59', 7),
        ('H60', 'H95', 8),
        ('I00', 'I99', 9),
        ('J00', 'J99', 10),
        ('K00', 'K93', 11),
        ('L00', 'L99', 12),
        ('M00', 'M99', 13),
        ('N00', 'N99', 14),
        ('O00', 'O99', 15),
        ('P00', 'P96', 16),
        ('Q00', 'Q99', 17),
        ('R00', 'R99', 18),
        ('S00', 'T98', 19),
        ('V01', 'Y98', 20),
        ('Z00', 'Z99', 21),
        ('U00', 'U99', 22),
    ]
    chapter_code = code[:3]
    for start, end, chapter_num in chapters:
        if start <= chapter_code <= end:
            return chapter_num
    raise ValueError(f"ICD-10 code {code} has no matching chapter")
def map_icd10_to_block(icd10_code, level=None):
    # Normalize and map code to a broader ICD-10 block id
    code = icd10_code.strip().upper()
    # ICD-10 block ranges (coarse categories)
    categories = [
        ('A00', 'A09', 1),
        ('A15', 'A19', 2),
        ('A20', 'A28', 3),
        ('A30', 'A49', 4),
        ('A50', 'A64', 5),
        ('A65', 'A69', 6),
        ('A70', 'A74', 7),
        ('A75', 'A79', 8),
        ('A80', 'A89', 9),
        ('A90', 'A99', 10),
        ('B00', 'B09', 11),
        ('B15', 'B19', 12),
        ('B20', 'B24', 13),
        ('B25', 'B34', 14),
        ('B35', 'B49', 15),
        ('B50', 'B64', 16),
        ('B65', 'B83', 17),
        ('B85', 'B89', 18),
        ('B90', 'B94', 19),
        ('B95', 'B98', 20),
        ('B99', 'B99', 21),
        ('C00', 'C97', 22),
        ('C00', 'C75', 23),
        ('C00', 'C14', 24),
        ('C15', 'C26', 25),
        ('C30', 'C39', 26),
        ('C40', 'C41', 27),
        ('C43', 'C44', 28),
        ('C45', 'C49', 29),
        ('C50', 'C50', 30),
        ('C51', 'C58', 31),
        ('C60', 'C63', 32),
        ('C64', 'C68', 33),
        ('C69', 'C72', 34),
        ('C73', 'C75', 35),
        ('C76', 'C80', 36),
        ('C81', 'C96', 37),
        ('C97', 'C97', 38),
        ('D00', 'D09', 39),
        ('D10', 'D36', 40),
        ('D37', 'D48', 41),
        ('D50', 'D53', 42),
        ('D55', 'D59', 43),
        ('D60', 'D64', 44),
        ('D65', 'D69', 45),
        ('D70', 'D77', 46),
        ('D80', 'D89', 47),
        ('E00', 'E07', 48),
        ('E10', 'E14', 49),

        ('E15', 'E16', 50),
        ('E20', 'E35', 51),
        ('E40', 'E46', 52),
        ('E50', 'E64', 53),
        ('E65', 'E68', 54),
        ('E70', 'E90', 55),
        ('F00', 'F09', 56),
        ('F10', 'F19', 57),
        ('F20', 'F29', 58),
        ('F30', 'F39', 59),
        ('F40', 'F48', 60),
        ('F50', 'F59', 61),
        ('F60', 'F69', 62),
        ('F70', 'F79', 63),
        ('F80', 'F89', 64),
        ('F90', 'F98', 65),
        ('F99', 'F99', 66),
        ('G00', 'G09', 67),
        ('G10', 'G14', 68),
        ('G20', 'G26', 69),
        ('G30', 'G32', 70),
        ('G35', 'G37', 71),
        ('G40', 'G47', 72),
        ('G50', 'G59', 73),
        ('G60', 'G64', 74),
        ('G70', 'G73', 75),
        ('G80', 'G83', 76),
        ('G90', 'G99', 77),
        ('H00', 'H06', 78),
        ('H10', 'H13', 79),
        ('H15', 'H22', 80),
        ('H25', 'H28', 81),
        ('H30', 'H36', 82),
        ('H40', 'H42', 83),
        ('H43', 'H45', 84),
        ('H46', 'H48', 85),
        ('H49', 'H52', 86),
        ('H53', 'H54', 87),
        ('H55', 'H59', 88),
        ('H60', 'H62', 89),
        ('H65', 'H75', 90),
        ('H80', 'H83', 91),
        ('H90', 'H95', 92),
        ('I00', 'I02', 93),
        ('I05', 'I09', 94),
        ('I10', 'I15', 95),
        ('I20', 'I25', 96),
        ('I26', 'I28', 97),
        ('I30', 'I52', 98),
        ('I60', 'I69', 99),

        ('I70', 'I79', 100),
        ('I80', 'I89', 101),
        ('I95', 'I99', 102),
        ('J00', 'J99', 103),
        ('J00', 'J06', 104),
        ('J09', 'J18', 105),
        ('J20', 'J22', 106),
        ('J30', 'J39', 107),
        ('J40', 'J47', 108),
        ('J60', 'J70', 109),
        ('J80', 'J84', 110),
        ('J85', 'J86', 111),
        ('J90', 'J94', 112),
        ('J95', 'J99', 113),
        ('K00', 'K14', 114),
        ('K20', 'K31', 115),
        ('K35', 'K38', 116),
        ('K40', 'K46', 117),
        ('K50', 'K52', 118),
        ('K55', 'K64', 119),
        ('K65', 'K67', 120),
        ('K70', 'K77', 121),
        ('K80', 'K87', 122),
        ('K90', 'K93', 123),
        ('L00', 'L08', 124),
        ('L10', 'L14', 125),
        ('L20', 'L30', 126),
        ('L40', 'L45', 127),
        ('L50', 'L54', 128),
        ('L55', 'L59', 129),
        ('L60', 'L75', 130),
        ('L80', 'L99', 131),
        ('M00', 'M25', 132),
        ('M00', 'M03', 133),
        ('M05', 'M14', 134),
        ('M15', 'M19', 135),
        ('M20', 'M25', 136),
        ('M30', 'M36', 137),
        ('M40', 'M54', 138),
        ('M40', 'M43', 139),
        ('M45', 'M49', 140),
        ('M50', 'M54', 141),
        ('M60', 'M79', 142),
        ('M60', 'M63', 143),
        ('M65', 'M68', 144),
        ('M70', 'M79', 145),
        ('M80', 'M94', 146),
        ('M80', 'M85', 147),
        ('M86', 'M90', 148),
        ('M91', 'M94', 149),

        ('M95', 'M99', 150),
        ('N00', 'N08', 151),
        ('N10', 'N16', 152),
        ('N17', 'N19', 153),
        ('N20', 'N23', 154),
        ('N25', 'N29', 155),
        ('N30', 'N39', 156),
        ('N40', 'N51', 157),
        ('N60', 'N64', 158),
        ('N70', 'N77', 159),
        ('N80', 'N98', 160),
        ('N99', 'N99', 161),
        ('O00', 'O08', 162),
        ('O10', 'O16', 163),
        ('O20', 'O29', 164),
        ('O30', 'O48', 165),
        ('O60', 'O75', 166),
        ('O80', 'O84', 167),
        ('O85', 'O92', 168),
        ('O94', 'O99', 169),
        ('P00', 'P04', 170),
        ('P05', 'P08', 171),
        ('P10', 'P15', 172),
        ('P20', 'P29', 173),
        ('P35', 'P39', 174),
        ('P50', 'P61', 175),
        ('P70', 'P74', 176),
        ('P75', 'P78', 177),
        ('P80', 'P83', 178),
        ('P90', 'P96', 179),
        ('Q00', 'Q07', 180),
        ('Q10', 'Q18', 181),
        ('Q20', 'Q28', 182),
        ('Q30', 'Q34', 183),
        ('Q35', 'Q37', 184),
        ('Q38', 'Q45', 185),
        ('Q50', 'Q56', 186),
        ('Q60', 'Q64', 187),
        ('Q65', 'Q79', 188),
        ('Q80', 'Q89', 189),
        ('Q90', 'Q99', 190),
        ('R00', 'R09', 191),
        ('R10', 'R19', 192),
        ('R20', 'R23', 193),
        ('R25', 'R29', 194),
        ('R30', 'R39', 195),
        ('R40', 'R46', 196),
        ('R47', 'R49', 197),
        ('R50', 'R69', 198),

        ('R70', 'R79', 199),
        ('R80', 'R82', 200),
        ('R83', 'R89', 201),
        ('R90', 'R94', 202),
        ('R95', 'R99', 203),
        ('S00', 'S09', 204),
        ('S10', 'S19', 205),
        ('S20', 'S29', 206),
        ('S30', 'S39', 207),
        ('S40', 'S49', 208),
        ('S50', 'S59', 209),
        ('S60', 'S69', 210),
        ('S70', 'S79', 211),
        ('S80', 'S89', 212),
        ('S90', 'S99', 213),
        ('T00', 'T07', 214),
        ('T08', 'T14', 215),
        ('T15', 'T19', 216),
        ('T20', 'T32', 217),
        ('T20', 'T25', 218),
        ('T26', 'T28', 219),
        ('T29', 'T32', 220),
        ('T33', 'T35', 221),
        ('T36', 'T50', 222),
        ('T51', 'T65', 223),
        ('T66', 'T78', 224),
        ('T79', 'T79', 225),
        ('T80', 'T88', 226),
        ('T90', 'T98', 227),
        ('V01', 'X59', 228),
        ('V01', 'V99', 229),
        ('V01', 'V09', 230),
        ('V10', 'V19', 231),
        ('V20', 'V29', 232),
        ('V30', 'V39', 233),
        ('V40', 'V49', 234),
        ('V50', 'V59', 235),
        ('V60', 'V69', 236),
        ('V70', 'V79', 237),
        ('V80', 'V89', 238),
        ('V90', 'V94', 239),
        ('V95', 'V97', 240),
        ('V98', 'V99', 241),
        ('W00', 'W19', 242),
        ('W20', 'W49', 243),
        ('W50', 'W64', 244),
        ('W65', 'W74', 245),
        ('W75', 'W84', 246),
        ('W85', 'W99', 247),
        ('X00', 'X09', 248),
        ('X10', 'X19', 249),
        ('X20', 'X29', 250),
        ('X30', 'X39', 251),
        ('X40', 'X49', 252),
        ('X50', 'X57', 253),
        ('X58', 'X59', 254),
        ('X60', 'X84', 255),
        ('X85', 'Y09', 256),
        ('Y10', 'Y34', 257),
        ('Y35', 'Y36', 258),
        ('Y40', 'Y84', 259),
        ('Y40', 'Y59', 260),
        ('Y60', 'Y69', 261),
        ('Y70', 'Y82', 262),
        ('Y83', 'Y84', 263),
        ('Y85', 'Y89', 264),
        ('Y90', 'Y98', 265),
        ('Z00', 'Z13', 266),
        ('Z20', 'Z29', 267),
        ('Z30', 'Z39', 268),
        ('Z40', 'Z54', 269),
        ('Z55', 'Z65', 270),
        ('Z70', 'Z76', 271),
        ('Z80', 'Z99', 272),
        ('U00', 'U79', 273),
        ('U80', 'U89', 274),
    ]


    # Extract the first three chars as block key
    block_code = code[:3]

    # Find matching block range
    for start, end, chapter_num in categories:
        if start <= block_code <= end:
            return chapter_num

    # No matching block
    raise ValueError(f"ICD-10 code {code} has no matching block")




# Build category->id mappings for each level
category_to_id_all_levels = {}

for level in range(1, LEVEL + 1):
    if level == 1:
        get_category = map_icd10_to_chapter
    elif level == 2:
        get_category = map_icd10_to_block
    else:
        get_category = get_icd10_category

    # Collect and sort all categories present at this level
    all_disease_categories = sorted(
        {get_category(disease,level) for diseases in date_dict.values() for disease in diseases})

    # Assign an incremental id to each category
    category_to_id = {category: idx + 1 for idx, category in enumerate(all_disease_categories)}
    category_to_id_all_levels[level] = category_to_id

    print(f"Disease categories and IDs for level {level}:")
    for category, category_id in category_to_id.items():
        print(f"Category: {category}, ID: {category_id}")

# Convert disease lists into id lists per patient (one sublist per level)
date_dict_with_ids = {}
for patient_id, diseases in date_dict.items():
    patient_diseases = [[] for _ in range(LEVEL)]
    for disease in diseases:
        for level in range(1, LEVEL + 1):
            if level == 1:
                get_category = map_icd10_to_chapter
            elif level == 2:
                get_category = map_icd10_to_block
            else:
                get_category = get_icd10_category

            category = get_category(disease, level)
            patient_diseases[level - 1].append(category_to_id_all_levels[level][category])
    date_dict_with_ids[patient_id] = patient_diseases



# Print final mapping summary for the last processed level
print("Disease categories and their IDs:")
for category, category_id in category_to_id.items():
    print(f"Category: {category}, ID: {category_id}")

# Save category->id mapping
with open(f"../{DATASET_DIR}/dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as f:
    pickle.dump(category_to_id, f)

# Save per-patient lists of category ids
with open(f"../{DATASET_DIR}/patient_dis2id_{EXCLUDE_DAY}_{'_'.join(TARGET_PATTERN)}_{LEVEL}.pkl", "wb") as f:
    pickle.dump(date_dict_with_ids, f)

