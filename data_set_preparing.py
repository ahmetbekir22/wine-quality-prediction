# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd


# from imblearn.over_sampling import SMOTE
# from collections import Counter

# white_wine_path = 'wine_quality_data/winequality-white.csv'
# red_wine_path = 'wine_quality_data/winequality-red.csv'

# white_wine_data = pd.read_csv(white_wine_path, sep=';')
# red_wine_data = pd.read_csv(red_wine_path, sep=';')

# # Separate features and target for red and white wines
# X_white = white_wine_data.drop(columns=['quality'])
# y_white = white_wine_data['quality']
# X_red = red_wine_data.drop(columns=['quality'])
# y_red = red_wine_data['quality']

# # Split data into training and testing sets (80%-20% split)
# X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(
#     X_white, y_white, test_size=0.2, random_state=42, stratify=y_white
# )

# X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
#     X_red, y_red, test_size=0.2, random_state=42, stratify=y_red
# )

# # Scale features using StandardScaler
# scaler_white = StandardScaler()
# scaler_red = StandardScaler()

# X_white_train_scaled = scaler_white.fit_transform(X_white_train)
# X_white_test_scaled = scaler_white.transform(X_white_test)

# X_red_train_scaled = scaler_red.fit_transform(X_red_train)
# X_red_test_scaled = scaler_red.transform(X_red_test)

# # Verify shapes of training and testing sets
# print("White Wine - Training Set Shape:", X_white_train_scaled.shape)
# print("White Wine - Testing Set Shape:", X_white_test_scaled.shape)
# # print("Red Wine - Training Set Shape:", X_red_train_scaled.shape)
# # print("Red Wine - Testing Set Shape:", X_red_test_scaled.shape)

# # Check original class distributions
# print("Original White Wine Class Distribution:", Counter(y_white_train))
# #print("Original Red Wine Class Distribution:", Counter(y_red_train))

# # Adjust SMOTE's k_neighbors parameter if necessary
# smote_white = SMOTE(random_state=42, k_neighbors=3)  # Adjust k_neighbors based on the smallest class size
# X_white_train_balanced, y_white_train_balanced = smote_white.fit_resample(
#     X_white_train_scaled, y_white_train
# )

# smote_red = SMOTE(random_state=42, k_neighbors=3)  # Adjust k_neighbors based on the smallest class size
# X_red_train_balanced, y_red_train_balanced = smote_red.fit_resample(
#     X_red_train_scaled, y_red_train
# )

# # Check the new class distributions
# white_class_distribution = Counter(y_white_train_balanced)
# red_class_distribution = Counter(y_red_train_balanced)

# print("White Wine Class Distribution (After SMOTE):", white_class_distribution)
# #print("Red Wine Class Distribution (After SMOTE):", red_class_distribution)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Veri yükleme
white_wine_path = 'wine_quality_data/winequality-white.csv'
red_wine_path = 'wine_quality_data/winequality-red.csv'

white_wine_data = pd.read_csv(white_wine_path, sep=';')
red_wine_data = pd.read_csv(red_wine_path, sep=';')

# Özellikler ve hedef değişkeni ayırma
X_white = white_wine_data.drop(columns=['quality'])
y_white = white_wine_data['quality']
X_red = red_wine_data.drop(columns=['quality'])
y_red = red_wine_data['quality']

# Veriyi eğitim ve test setlerine ayırma
X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(
    X_white, y_white, test_size=0.2, random_state=42, stratify=y_white
)

X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
    X_red, y_red, test_size=0.2, random_state=42, stratify=y_red
)

# Özellikleri ölçeklendirme
scaler_white = StandardScaler()
scaler_red = StandardScaler()

X_white_train_scaled = scaler_white.fit_transform(X_white_train)
X_white_test_scaled = scaler_white.transform(X_white_test)

X_red_train_scaled = scaler_red.fit_transform(X_red_train)
X_red_test_scaled = scaler_red.transform(X_red_test)

# SMOTE ile eğitim verisini dengeliyoruz
smote_white = SMOTE(random_state=42, k_neighbors=7)
X_white_train_balanced, y_white_train_balanced = smote_white.fit_resample(X_white_train_scaled, y_white_train)

smote_red = SMOTE(random_state=42, k_neighbors=3)
X_red_train_balanced, y_red_train_balanced = smote_red.fit_resample(X_red_train_scaled, y_red_train)

# Yeni sınıf dağılımlarını kontrol etme
white_class_distribution = Counter(y_white_train_balanced)
red_class_distribution = Counter(y_red_train_balanced)

print("White Wine Class Distribution (After SMOTE):", white_class_distribution)
