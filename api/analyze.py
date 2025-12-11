import os
import io
import json
import base64
import joblib
import librosa
import numpy as np
from pymongo import MongoClient

# --- متغيرات أساسية ---
# اسم المستخدم في MongoDB
DB_USER = "bee_hive_api"
# اسم قاعدة البيانات التي سننشئها
DB_NAME = "BeeHiveDB"
# اسم المجموعة (Collection) داخل القاعدة لحفظ السجلات
COLLECTION_NAME = "AnalysisLogs"

# تحميل النماذج (يجب أن تكون هذه الملفات في نفس مستوى ملف analyze.py)
try:
    rf_model = joblib.load('final_rf_model_tuned.pkl')
    mfcc_scaler = joblib.load('mfcc_scaler.pkl')
except Exception as e:
    # هذا الخطأ يحدث إذا لم يتم العثور على الملفات عند النشر
    print(f"Error loading models: {e}")
    rf_model = None
    mfcc_scaler = None

# --- ربط قاعدة البيانات MongoDB Atlas ---
# هذا الرابط سنخزنه كمتغير بيئة في Vercel بدلاً من وضعه مباشرة في الكود
MONGO_URI = os.environ.get('MONGO_URI')

# دالة مساعدة لحفظ السجلات في قاعدة البيانات
def save_analysis_log(log_data):
    if not MONGO_URI:
        print("MongoDB URI not set. Skipping database save.")
        return

    try:
        # الاتصال بقاعدة البيانات
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # إضافة السجل مع وقت وتاريخ التحليل
        log_data['timestamp'] = datetime.datetime.utcnow() 
        
        # حفظ السجل
        collection.insert_one(log_data)
        client.close()
        print("Analysis log saved successfully to MongoDB.")
        
    except Exception as e:
        print(f"Error connecting or saving to MongoDB: {e}")
        # يجب التأكد من وجود client وإغلاقه في حالة الفشل
        if 'client' in locals():
            client.close()


def analyze_audio(request):
    """
    الدالة الرئيسية التي تستدعيها Vercel Function.
    تستقبل بيانات الصوت، تحللها، وتحفظ النتيجة في MongoDB.
    """
    if request.method != 'POST':
        return json.dumps({"error": "Only POST requests are accepted"}), 405

    try:
        # 1. استلام البيانات
        request_json = request.get_json(silent=True)
        if not request_json or 'audio_data' not in request_json:
            raise ValueError("Missing 'audio_data' in request body.")

        audio_b64 = request_json['audio_data']
        # فك ترميز Base64 إلى بيانات ثنائية
        audio_bytes = base64.b64decode(audio_b64)
        
        # 2. إعداد البيانات للتحليل
        # استخدام io.BytesIO لإنشاء ملف صوتي وهمي في الذاكرة
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=None) 
        
        # 3. استخراج الميزات (MFCCs)
        # تأكد من أن librosa حملت البيانات بنجاح
        if y is None or sr is None:
             raise ValueError("Librosa failed to load audio data.")
             
        # استخدام نفس معاملات الاستخراج والقياس المستخدمة في التدريب
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=128, n_fft=512)
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
        
        # 4. قياس البيانات وتصنيفها
        scaled_features = mfcc_scaler.transform(mfccs_mean)
        prediction_index = rf_model.predict(scaled_features)[0]

        # 5. تعيين النتيجة (يجب مطابقة هذه الفهارس مع فهارس تدريب النموذج)
        behavior_map = {
            0: "Normal",
            1: "Swarming",
            2: "Queen Absence",
            3: "Disease"
        }
        prediction_label = behavior_map.get(prediction_index, "Unknown")
        
        # 6. إعداد السجل للحفظ في MongoDB
        log_data = {
            "predicted_index": int(prediction_index),
            "behavior_label": prediction_label,
            "source_device": request_json.get('device_id', 'ESP32_Unknown'),
            # يمكن إضافة بيانات أخرى مثل درجة الحرارة هنا
        }
        
        # 7. حفظ السجل في قاعدة البيانات
        save_analysis_log(log_data)

        # 8. إرسال الرد إلى جهاز ESP32/A9G
        response_data = {
            "status": "success",
            "prediction": prediction_label,
            "timestamp": log_data['timestamp'].isoformat() # إرسال الوقت في الرد أيضاً
        }
        
        return json.dumps(response_data), 200

    except Exception as e:
        error_message = f"Processing Error: {str(e)}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message}), 500