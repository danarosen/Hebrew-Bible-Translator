# A script to generate plots for the 6.806 Final Paper.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plot 1: Training loss evolution. Plot Test Loss and Training Loss over epoch.
# Show Loss for train and test sets 
"""Discussion points:
1. The nature of the data in the training set and the test set is similar (As Zach showed). This is reflected in the resutlts.
2. Write about how many words we have in our vocabulary. Number of sentences should be detailed in the dataset section.
3. Performance degrades if going through less training -- Some training is actually enough.
4. With nikud added in, show that performance degrades.
5. Show examples of word without nikkud and word with nikud, and then show the unicode for the words. Discuss how each menukad word is unique.
5a. Plot distribution of words (histogram, type).
6. Convergence is harder to achieve. But could more nuance be captured?
"""


####################
# PLOT LOSS EVOLUTION End-To-End Transformer.
####################
train_loss_per_token_40_epochs_no_nikkud =  np.array([8.098544120788574, 7.371527194976807, 6.843085289001465, 6.576930046081543, 6.490429878234863, 6.344634532928467, 6.100920677185059, 5.739053249359131, 5.405462265014648, 5.001719951629639, 4.578732967376709, 4.213589191436768, 3.8373141288757324, 3.343641519546509, 2.9117727279663086, 2.5898962020874023, 2.1500778198242188, 1.7482235431671143, 1.4291898012161255, 1.1625947952270508, 0.9552983641624451, 0.7544848322868347, 0.6238949298858643, 0.5515583753585815, 0.5021371841430664, 0.45032185316085815, 0.43675851821899414, 0.4040485918521881, 0.37284842133522034, 0.35043904185295105, 0.31745001673698425, 0.29175809025764465, 0.2778208553791046, 0.26275303959846497, 0.24529269337654114, 0.23393718898296356, 0.224114790558815, 0.21264834702014923, 0.20124097168445587, 0.19657033681869507])

test_loss_per_token_40_epochs_no_nikkud = np.array([7.7001, 7.0657, 6.7092, 6.4664, 6.2574, 6.0245, 5.6869, 5.2533, 4.849, 4.3897, 3.8981, 3.4195, 2.9512, 2.4723, 2.0385, 1.6358, 1.2633, 0.9698, 0.7406, 0.5789, 0.4687, 0.404, 0.3783, 0.3593, 0.3336, 0.3336, 0.3225, 0.3014, 0.2841, 0.2714, 0.2423, 0.2334, 0.2221, 0.2158, 0.2014, 0.1908, 0.1848, 0.1763, 0.1691, 0.164])

train_loss_per_token_40_epochs_yes_nikkud = np.array([8.2042, 7.4985, 6.8925, 6.7107, 6.6097, 6.3585, 6.1111, 5.8443, 5.6209, 5.2518, 4.8943, 4.4769, 3.9485, 3.6218, 3.2218, 2.8331, 2.3468, 2.0065, 1.6587, 1.291, 1.0737, 0.8606, 0.7271, 0.6153, 0.548, 0.5062, 0.4705, 0.4391, 0.3904, 0.3574, 0.3317, 0.3007, 0.2876, 0.2693, 0.2523, 0.2434, 0.2294, 0.2115, 0.2053, 0.1984])

test_loss_per_token_40_epochs_yes_nikkud = np.array([7.8213, 7.1851, 6.7371, 6.5782, 6.4329, 6.0502, 5.6678, 5.3152, 4.9431, 4.5429, 4.1274, 3.6461, 3.1383, 2.6958, 2.2366, 1.8082, 1.4178, 1.1076, 0.8406, 0.6554, 0.5342, 0.4722, 0.4283, 0.3952, 0.3829, 0.3574, 0.3389, 0.3118, 0.2893, 0.2727, 0.2549, 0.2401, 0.2254, 0.2172, 0.2071, 0.1972, 0.1849, 0.1768, 0.173, 0.1647])

plt.figure(0)
plt.plot(train_loss_per_token_40_epochs_no_nikkud, label="Train Set Average Loss ", color = "red", linestyle= "dashdot")
plt.plot(test_loss_per_token_40_epochs_no_nikkud, label="Validation Set Average Loss ", color = "red")

plt.plot(train_loss_per_token_40_epochs_yes_nikkud, label= "Train Set Average Loss  with nikud", marker = ".", linestyle= "dashdot", color = 'blue')
plt.plot(test_loss_per_token_40_epochs_yes_nikkud, label= "Validation Set Average Loss  with nikud", marker = ".", color = 'blue')
plt.legend()
plt.title("End-To-End Transformer\nEvolution of Average Loss")
plt.ylabel("Average Loss")
plt.xlabel("Epoch Number")
plt.show()

# print([float(i[30:][i[30:].find("tensor(")+7: i[30:].find("tensor(") + 13]) for i in  a.split("\n")])
# print([float(i[:][i[:].find("tensor(")+7: i[:].find("tensor(") + 13]) for i in  a.split("\n")])

####################
# PLOT BLEU EVOLUTION ETE Transformer.
####################

train_bleu_per_sample_31_epochs_no_nikkud = [0, 0, 0, 0, 0, 0.7441860465116279, 2.179401993355482, 1.2192691029900333, 3.2691029900332227, 4.504983388704319, 3.2325581395348837, 4.45514950166113, 8.09966777408638, 8.249169435215947, 18.737541528239202, 27.182724252491695, 45.86046511627907, 50.87375415282392, 65.68106312292359, 75.8671096345515, 81.40199335548172, 84.52159468438538, 90.27242524916943, 89.0299003322259, 90.30564784053156, 92.82392026578073, 93.80066445182725, 96.16611295681064, 96.35548172757476, 95.83720930232558, 97.04318936877077]


test_bleu_per_sample_31_epochs_no_nikkud = [0, 0, 0, 0, 0, 1.0664451827242525, 1.7475083056478404, 1.920265780730897, 1.5282392026578073, 1.9833887043189369, 2.990033222591362, 5.803986710963455, 5.657807308970099, 7.4186046511627906, 15.289036544850498, 29.308970099667775, 40.01328903654485, 53.30232558139535, 63.19601328903654, 69.7109634551495, 73.9734219269103, 78.265780730897, 79.53488372093024, 81.90697674418605, 82.37541528239203, 83.43853820598007, 86.96013289036544, 85.50830564784053, 88.29235880398672, 90.28571428571429, 92.71428571428571]

test_bleu_per_sample_31_epochs_yes_nikkud = [0, 0, 0, 0, 0, 0.009966777408637873, 1.0564784053156147, 1.435215946843854, 1.435215946843854, 2.046511627906977, 1.0166112956810631, 2.2059800664451825, 1.372093023255814, 2.877076411960133, 4.2425249169435215, 7.498338870431894, 9.362126245847175, 12.116279069767442, 21.182724252491695, 34.83720930232558, 43.63787375415282, 49.11295681063123, 56.79734219269103, 63.199335548172755, 76.25249169435216, 72.12956810631229, 81.11295681063123, 83.93687707641196, 87.00332225913621, 90.39202657807309, 90.3421926910299, 91.71428571428571, 93.10631229235881, 92.8172757475083, 95.80730897009967, 95.19601328903654, 95.86046511627907, 97.4485049833887, 98.17607973421927] 

train_bleu_score_yes_nikud = [0, 0, 0, 0, 0, 0.04318936877076412, 1.7242524916943522, 1.1362126245847175, 1.4186046511627908, 1.1096345514950166, 1.5249169435215948, 2.0199335548172757, 2.574750830564784, 3.4119601328903655, 3.338870431893688, 5.764119601328904, 10.192691029900333, 15.561461794019934, 27.58139534883721, 38.182724252491695, 50.388704318936874, 57.27574750830565, 66.15614617940199, 79.08305647840531, 80.50830564784053, 83.03654485049834, 90.05647840531562, 91.69102990033223, 94.734219269103, 94.62458471760797, 94.88704318936877, 95.94019933554817, 96.61461794019934, 97.32558139534883, 96.265780730897, 97.89368770764119, 98.46511627906976, 98.60132890365449, 98.30897009966777]

plt.figure(0)
# No nikud.
# plt.plot(train_bleu_per_sample_31_epochs_no_nikkud, label="Train Set Average BLEU Score", color = "red", linestyle="dashdot")
plt.plot(test_bleu_per_sample_31_epochs_no_nikkud, label="Validation Set Average BLEU Score", color = "red")
# Yes nikud.
plt.plot(test_bleu_per_sample_31_epochs_yes_nikkud, label="Validation Set Average BLEU Score with nikud", marker = ".", color = "blue")

# plt.plot(train_bleu_score_yes_nikud, label="Train Set Average BLEU Score with nikud", marker = ".", color = "blue", linestyle="dashdot")
# plt.plot(train_bleu_score_no_nikud)

plt.legend()
plt.title("End-To-End Transformer\nEvolution of Average BLEU Score")
plt.ylabel("Average BLEU Score")
plt.xlabel("Epoch Number")
plt.ylim(0,100)
plt.show()


####################
# PLOT LOSS EVOLUTION Fine-Tuned Multilingual Transformer.
####################

train_loss_per_epoch_no_nikkud = np.array([2.09, 1.4, 1.03, 0.77, 0.58954, 0.4400, 0.3348, 0.25202, 0.1925, 0.15188])

train_loss_per_epoch_yes_nikkud = np.array([1.05, 0.716779, 0.57706,  0.48, 0.4, 0.3414, 0.2882, 0.2427, 0.2034, 0.17227])

test_loss_per_token_40_epochs_yes_nikkud = np.array([])

plt.figure(0)
plt.plot(train_loss_per_epoch_no_nikkud, label="Train Set Average Loss", color = "red", linestyle= "dashdot")
# plt.plot(test_loss_per_token_40_epochs_no_nikkud, label="Validation Set Average Loss ", color = "red")

plt.plot(train_loss_per_epoch_yes_nikkud, label="Train Set Average Loss with nikud", color = 'blue', marker = ".", linestyle= "dashdot")
# plt.plot(test_loss_per_token_40_epochs_yes_nikkud, label="Validation Set Average Loss  with nikud", marker = ".", linestyle = "dashdot", color = 'blue')

plt.legend()
plt.title("Fine-Tuned Multilingual Transformer\nEvolution of Average Loss")
plt.ylabel("Average Loss")
plt.xlabel("Epoch Number")
plt.ylim(0, None)
plt.show()




####################
# PLOT BLEU EVOLUTION FTM Transformer.
####################

# First entry is without training.
test_bleu_no_nikkud_ftmt = np.array([0.50, 0.61, 0.65, 0.67, 0.69, 0.71955, 0.73831, 0.760705, 0.774, 0.7798, 0.784320])*100
test_bleu_yes_nikkud_ftmt = np.array([0.15948, 0.730, 0.74908, 0.76, 0.77, 0.7812, 0.789, 0.801, 0.811, 0.811, 0.82])*100



plt.figure(0)
plt.plot(test_bleu_no_nikkud_ftmt, label="Validation Set Average BLEU Score", color = "red")
plt.plot(test_bleu_yes_nikkud_ftmt, label="Validation Set Average BLEU Score Nikud", marker = ".", color = "blue")

plt.xticks([i for i in range(len(test_bleu_no_nikkud_ftmt))], [i-1 for i in range(len(test_bleu_no_nikkud_ftmt))])

# plt.plot((test_loss_per_token_40_epochs_no_nikkud * 100.0 / test_loss_per_token_40_epochs_no_nikkud.max())[:31], label="Validation Set Average Loss ", marker = ".", color = "red")


plt.legend()
plt.title("Fine-Tuned Multilingual Transformer\nEvolution of Average BLEU Score")
plt.ylabel("Average BLEU Score")
plt.xlabel("Epoch Number")
plt.ylim(0,100)
plt.show()

'''

בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ
בראשית ברא אלהים את השמים ואת הארץ

הִנָּךְ יָפָה רַעְיָתִי, הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים.

וְצִפִּיתָ אֹתוֹ, זָהָב טָהוֹר וְעָשִׂיתָ לּוֹ זֵר זָהָב, סָבִיב

דוד אוהב לאכול הרבה אוכל טעים

Comparison to nikud/nikud in model and in input text.
'''


'''
##########
FTMT Nikud:
##########

Orignal
בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ
RESULT
הֶסְכֵּם אֶת בָּרָא הוּא אֱלֹהִים וְאֶת הַשָּׁמַיִם וְאֶת הָאָרֶץ

Orignal
בראשית ברא אלהים את השמים ואת הארץ
RESULT
בְּבַקָּשָׁה בָּהֶם אֱלִים הָיוּ וְהַשָּׁם

Orignal
הִנָּךְ יָפָה רַעְיָתִי, הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים.
RESULT
רַעְיָתִי אֶת יָפָה הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים

Orignal
הנך יפה רעיתי, הנך יפה עיניך יונים.
RESULT
מְקוֹמָם חֲבֵרוֹת חֲבֵרוֹת חֲבֵרוֹת חֲבֵרוֹת חֲבֵרוֹת

Orignal
וְצִפִּיתָ אֹתוֹ, זָהָב טָהוֹר וְעָשִׂיתָ לּוֹ זֵר זָהָב, סָבִיב
RESULT
וּתְצַפֶּה אוֹתוֹ מִזָּהָב טָהוֹר לוֹ וְתַעֲשֶׂה זָהָב זֵר מִסָּבִיב

Orignal
לַֽיהֹוָֽה נִזְבְּחָ֥ה נֵלְכָ֖ה אֹֽמְרִ֔ים אַתֶּ֣ם עַל כֵּן֙ נִרְפִּ֑ים אַתֶּ֖ם נִרְפִּ֥ים וַיֹּ֛אמֶר
RESULT
לַיהוה קָרְבָּנוֹת וְנַקְרִיב נֵלֵךְ אוֹמְרִים אַתֶּם לָכֵן בַּטְלָנִים בַּטְלָנִים אַתֶּם לָהֶם אָמַר

Orignal
ליהוה נזבחה נלכה אמרים אתם על כן נרפים אתם נרפים ויאמר
RESULT
לַיהוה נַקֵּל נַקֵּל כְּשֶׁאָמְרוּ נַקֵּל כְּשֶׁאָמְרוּ

Orignal
הָהָר הַגָּדוֹל הוּא הָהָר הֲכִי גָּבוֹהַּ
RESULT
הָהָר הַגָּדוֹל הוּא הָהָר הֲרֵי גָּבוֹהּ

Orignal
ההר הגדול הוא ההר הכי גבוה
RESULT
הַבֵּן בַּעַל הַבֵּן בַּעַל הוּא

Orignal
יֵשׁ לִי כֶּלֶב מָהִיר מְאוֹד
RESULT
כֶּלֶב לִי יֵשׁ מָהִיר מְאוֹד

Orignal
יש לי כלב מהיר מאוד
RESULT
בְּהֵמָה זֶה בְּכָךְ

##########
Fine-Tuned Multilingual Transformer
##########

Orignal
בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ
RESULT
הבהמות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן העולות באותן

Orignal
בראשית ברא אלהים את השמים ואת הארץ
RESULT
בהתחלה ברא אלהים השמים את הארץ ואת

Orignal
הִנָּךְ יָפָה רַעְיָתִי, הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים.
RESULT
יפאה היין שפיכת רעיונותי את יפוך היין שפיכת יפאה היין שפיכת את יפוך הוא שלך היין שפיכת

Orignal
הנך יפה רעיתי, הנך יפה עיניך יונים.
RESULT
יפה רעיתי יפה עיניך יונים

Orignal
וְצִפִּיתָ אֹתוֹ, זָהָב טָהוֹר וְעָשִׂיתָ לּוֹ זֵר זָהָב, סָבִיב
RESULT
וצבאיהם אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי אצלי

Orignal
לַֽיהֹוָֽה נִזְבְּחָ֥ה נֵלְכָ֖ה אֹֽמְרִ֔ים אַתֶּ֣ם עַל כֵּן֙ נִרְפִּ֑ים אַתֶּ֖ם נִרְפִּ֥ים וַיֹּ֛אמֶר
RESULT
ואלו בקודש יגרום שלא כדי תבואה נתחלקו ואלו בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש יגרום שלא כדי בקודש

Orignal
ליהוה נזבחה נלכה אמרים אתם על כן נרפים אתם נרפים ויאמר
RESULT
ליהוה קרבנות ונקריב נלך אומרים אתם לכן בטלנים בטלנים אתם להם אמר

Orignal
הָהָר הַגָּדוֹל הוּא הָהָר הֲכִי גָּבוֹהַּ
RESULT
הכהן הכהן הכהן הכהן הכהן הכהן הכהן

Orignal
ההר הגדול הוא ההר הכי גבוה
RESULT
הגדול ההר הוא הגבוהה ההר

Orignal
יֵשׁ לִי כֶּלֶב מָהִיר מְאוֹד
RESULT
יאבד שלא כדי השכיבה אבן עם מאויב שאינו מי

Orignal
יש לי כלב מהיר מאוד
RESULT
כלב לי יש מאוד מהיר


##########
ETET Plain
##########

Original
 בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ
Translation: None

Original
 בראשית ברא אלהים את השמים ואת הארץ
Pred Translation:	SOS לחלקים אותה ויחלק מהעור הארץ ואת הארץ ואת הארץ ואת הארץ ואת 

Original
 הִנָּךְ יָפָה רַעְיָתִי, הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים.
Translation: None

Original
 הנך יפה רעיתי, הנך יפה עיניך יונים.
Translation: None

Original
 וְצִפִּיתָ אֹתוֹ, זָהָב טָהוֹר וְעָשִׂיתָ לּוֹ זֵר זָהָב, סָבִיב
Translation: None

Original
 לַֽיהֹוָֽה נִזְבְּחָ֥ה נֵלְכָ֖ה אֹֽמְרִ֔ים אַתֶּ֣ם עַל כֵּן֙ נִרְפִּ֑ים אַתֶּ֖ם נִרְפִּ֥ים וַיֹּ֛אמֶר
Translation: None

Original
 ליהוה נזבחה נלכה אמרים אתם על כן נרפים אתם נרפים ויאמר
Pred Translation:	SOS ליהוה קרבנות ונקריב נלך אומרים אתם לכן בטלנים הם להם אמר 

Original
 הָהָר הַגָּדוֹל הוּא הָהָר הֲכִי גָּבוֹהַּ
Translation: None

Original
 ההר הגדול הוא ההר הכי גבוה
Translation: None

Original
 יֵשׁ לִי כֶּלֶב מָהִיר מְאוֹד
Translation: None

Original
 יש לי כלב מהיר מאוד
Translation: None

Original
 ממוכן כדבר המלך ויעש והשרים המלך בעיני הדבר וייטב
Pred Translation:	SOS ממוכן כדברי עשה והמלך והשרים המלך בעיני טוב נראה הרעיון 

##########
ETET Nikud
##########
Original
 בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ
Pred Translation:	SOS הַחַטָּאת אֶת פַּר אֶת וְיִשְׁחַט מִשְׁפַּחְתּוֹ וְעַל עַצְמוֹ עַל וִיכַפֵּר הָאָרֶץ אֶת יַגִּישׁ 

Original
 בראשית ברא אלהים את השמים ואת הארץ
Translation: None

Original
 הִנָּךְ יָפָה רַעְיָתִי, הִנָּךְ יָפָה עֵינַיִךְ יוֹנִים.
Translation: None

Original
 הנך יפה רעיתי, הנך יפה עיניך יונים.
Translation: None

Original
 וְצִפִּיתָ אֹתוֹ, זָהָב טָהוֹר וְעָשִׂיתָ לּוֹ זֵר זָהָב, סָבִיב
Translation: None

Original
 לַֽיהֹוָֽה נִזְבְּחָ֥ה נֵלְכָ֖ה אֹֽמְרִ֔ים אַתֶּ֣ם עַל כֵּן֙ נִרְפִּ֑ים אַתֶּ֖ם נִרְפִּ֥ים וַיֹּ֛אמֶר
Pred Translation:	SOS לַיהוה קָרְבָּנוֹת וְנַקְרִיב נֵלֵךְ אוֹמְרִים אַתֶּם לָכֵן בַּטְלָנִים בַּטְלָנִים הֵם כִּי בָּנִים בַּטְלָנִים הֵם 

Original
 ליהוה נזבחה נלכה אמרים אתם על כן נרפים אתם נרפים ויאמר
Translation: None

Original
 הָהָר הַגָּדוֹל הוּא הָהָר הֲכִי גָּבוֹהַּ
Translation: None

Original
 ההר הגדול הוא ההר הכי גבוה
Translation: None

Original
 יֵשׁ לִי כֶּלֶב מָהִיר מְאוֹד
Translation: None

Original
 יש לי כלב מהיר מאוד
Translation: None

Original
 ממוכן כדבר המלך ויעש והשרים המלך בעיני הדבר וייטב
Translation: None
'''
