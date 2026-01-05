#include <math.h>
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <arduinoFFT.h>
#include <WiFi.h>
#include <WebServer.h>


// Bắt buộc phải có file này trong thư mục

#include "model_data.h"


// 1. CẤU HÌNH CẢM BIẾN & PIN

// Hiệu chỉnh cảm biến
#define cambiendong 35
#define cambienap 34
#define scalev 1000.0  
#define scalei 8.95   
#define adc_res 4095
#define Vref 3.3
#define vdcoffset 1835
#define idcoffset 1822
#define SAMPLES 1024
#define SAMPLING_FREQUENCY 5120

// KHAI BÁO CẤU TRÚC LƯU DỮ LIỆU ĐO ĐẠC
struct doluong {
    float Vrms;
    float Irms;
    float Power;
    float bieukien;
    float cosphi;
    float harmonics[4]; // Lưu hài bậc 1, 3, 5, 7
};
// KHAI BÁO MẢNG ĐỂ LƯU CÁC GIÁ TRỊ U I TỨC THỜI ĐƯỢC ĐO LIÊN TỤC
    float vReal[SAMPLES];
    float iReal[SAMPLES];
    float iImag[SAMPLES];
    ArduinoFFT<float> FFT = ArduinoFFT<float>(iReal, iImag, SAMPLES, SAMPLING_FREQUENCY);


// 2. PARAMETERS CHO MODEL (QUAN TRỌNG)

const int kModelInputSize = 4;
const int kModelOutputSize = 9;
const char* DEVICE_NAMES[] = {
    "Den_led",  // 0
    "Laptop1",  // 1 
    "Macbook",  // 2
    "Quat_lv1", // 3
    "Quat_lv2", // 4      
    "Quat_lv3", // 5      
    "Quat_lv4", // 6  
    "ko tải",   // 7
    "iPhone"    // 8

};

// Mean và StdDev
const float SCALER_MEAN[kModelInputSize] = {
    21.345099f, 0.515257f, 0.096624f, 0.023443f
};

// MẢNG STDDEV
const float SCALER_STDDEV[kModelInputSize] = {
    10.892426f, 0.232382f, 0.048665f, 0.026228f
};

// KHAI BÁO BỘ NHỚ
// Kích thước 60KB để đảm bảo đủ chỗ
constexpr int kTensorArenaSize = 40 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// 3. HÀM KIỂM TRA FILE MODEL_DATA.H (MỚI) *CÓ CẦN KHÔNG

void verifyModelData() {
    Serial.println("\n-----------------------------------");
    Serial.println("[KIEM TRA] Dang doc file model_data.h ...");
    // 1. Kiểm tra biến có tồn tại không (nếu compile được thì chắc chắn có)
    // In ra độ dài mảng
    Serial.printf("-> Kich thuoc Model: %d bytes\n", mlp_model_optimized_tflite_len);
    if (mlp_model_optimized_tflite_len < 100) {
        Serial.println("-> CANH BAO: Model qua nho! Co the file bi loi.");
    }
    // 2. In thử 16 byte đầu tiên (Header của TFLite)
    // File TFLite chuẩn thường có chuỗi 'TFL3' ở byte thứ 4-7
    Serial.print("-> 16 Byte dau tien (Hex): ");
    for (int i = 0; i < 16; i++) {
        Serial.printf("%02X ", mlp_model_optimized_tflite[i]);
    }
    Serial.println();
    // Kiểm tra chữ ký 'TFL3'
    if (mlp_model_optimized_tflite[4] == 'T' &&
        mlp_model_optimized_tflite[5] == 'F' &&
        mlp_model_optimized_tflite[6] == 'L' &&
        mlp_model_optimized_tflite[7] == '3') {
        Serial.println("-> XAC NHAN: Day la file TFLite hop le (Signature TFL3 Found).");
    } else {
        Serial.println("-> CANH BAO: Khong thay chu ky TFL3. File co the bi hong hoac khong phai TFLite.");
    }
    Serial.println("-----------------------------------\n");
}



// 4. CÁC HÀM XỬ LÝ AI

float normalize(float value, int index) {
    float norm = (value - SCALER_MEAN[index]) / SCALER_STDDEV[index];
    if (isnan(norm) || isinf(norm)) return 0.0f;
    if (norm > 10.0f) norm = 10.0f;
    if (norm < -10.0f) norm = -10.0f;
    return norm;
}

bool setupTFLite() {
    model = tflite::GetModel(mlp_model_optimized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("LOI: Model version mismatch! Ban can regenerate file model.");
        return false;
    }
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("LOI: Khong the cap phat bo nho (AllocateTensors)!");
        return false;
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    // --- KIỂM TRA KIỂU DỮ LIỆU ---
    Serial.print("-> Kieu du lieu Input: ");
    if (input->type == kTfLiteFloat32) Serial.println("FLOAT32 (OK)");
    else if (input->type == kTfLiteInt8) Serial.println("INT8 (Can sua code!)");
    else Serial.printf("OTHER (%d)\n", input->type);
    Serial.printf("-> TFLite da san sang! Arena used: %d bytes\n", interpreter->arena_used_bytes());
    return true;
}

int predict(float Power, float cosphi, float H1, float H7) {
    Serial.println("\n--- DEBUG DU LIEU CHI TIET ---");

    // Cấu hình input MLP
    float features[4] = {
        Power, cosphi, H1, H7
    };

    // Kiểm tra input
    Serial.print("Input Tensor: ");
    bool input_error = false;
    for (int i = 0; i < 4; i++) {
        float val = normalize(features[i], i);
        // Nạp vào input của model
        input->data.f[i] = val;
        // In ra để kiểm tra
        Serial.printf("[%.2f] ", val);
        // Kiểm tra xem có số nào bị hỏng (NaN hoặc Inf) không
        if (isnan(val) || isinf(val)) {
            input_error = true;
        }
    }
    Serial.println();
    if (input_error) {
        Serial.println("=> LOI: Phat hien Input bi NaN hoac Inf! Kiem tra lai cam bien/tinh toan.");
        return -1;
    }


    // 3. CHẠY MODEL

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("=> LOI: Interpreter Invoke failed!");
        return -1;
    }
    // 4. KIỂM TRA OUTPU
    float p0 = output->data.f[0];
    float p1 = output->data.f[1];
    float p2 = output->data.f[2];
    float p3 = output->data.f[3];
    float p4 = output->data.f[4];
    float p5 = output->data.f[5];
    float p6 = output->data.f[6];
    float p7 = output->data.f[7];
    float p8 = output->data.f[8];
    Serial.printf("Output Tensor Den_led:%.2f | Laptop1:%.2f | Macbook:%.2f | quat_lv1:%.2f| quat_lv2:%.2f|quat_lv3:%.2f|quat_lv4:%.2f|ko tai:%.2f |iphone:%.2f\n", p0, p1, p2, p3, p4, p5, p6, p7, p8);
    if (isnan(p0) || isnan(p1)) {
        Serial.println("=> KET LUAN: Input ngon lanh nhung Output van NaN -> Loi tai MODEL hoac LIBRARY.");
        return -1;
    }
    // Tìm kết quả tốt nhất
    int bestClass = 0;
    float bestProb = p0;
    for (int i = 1; i < kModelOutputSize; i++) {
        if (output->data.f[i] > bestProb) {
            bestProb = output->data.f[i];
            bestClass = i;
        }
    }
    return bestClass;
}



// 5. ĐỌC CẢM BIẾN

doluong Calc() {
    unsigned long sampling_period_us = round(1000000 * (1.0 / SAMPLING_FREQUENCY));
    unsigned long microseconds = micros();
    doluong Result = {0};

    // Lấy dữ liệu liên tục và mẫu cách đều 1024 mẫu với đúng tần số 5120 Hz
    for (int i = 0; i < SAMPLES; i++) {
        vReal[i] = analogRead(cambienap);
        iReal[i] = analogRead(cambiendong);
        iImag[i] = 0;
        //Fixed period sampling
        while (micros() - microseconds < sampling_period_us){
            //Busy wait
        }
        microseconds += sampling_period_us;
    }

    // Dùng FFT để loại bỏ thành phần offset
    FFT.dcRemoval(vReal, SAMPLES); 
    FFT.dcRemoval(iReal, SAMPLES); 

    // Tính toán RMS và Power (Lúc này vReal và iReal đã sạch DC)
    double v_sum = 0, i_sum = 0, p_sum = 0;
    for (int i = 0; i < SAMPLES; i++){
        v_sum += (double)vReal[i] * vReal[i];
        i_sum += (double)iReal[i] * iReal[i];
        p_sum += (double)vReal[i] * iReal[i];
    }
   
    // Tính toán P,S,cosphi chuyển về giá trị V I thật
    float v_factor = (Vref / (float)adc_res) * scalev;
    float i_factor = (Vref / (float)adc_res) * scalei;

    Result.Vrms = sqrt(v_sum/(float)SAMPLES) * v_factor;
    Result.Irms = sqrt(i_sum/(float)SAMPLES) * i_factor;
    Result.Power = (p_sum/(float)SAMPLES) * v_factor * i_factor;
    Result.bieukien = Result.Vrms * Result.Irms;
    Result.cosphi = (Result.bieukien > 0) ? (Result.Power / Result.bieukien) : 0;

    // Tính toán FFT
    FFT.dcRemoval(); // Tự hiểu là iReal
    FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();

    // 2. Hệ số hiệu chỉnh (Tính một lần để tiết kiệm CPU)
    // 1.85 là bù cửa sổ Hamming, 1.414 là đổi Peak sang RMS
    float calibration_factor = (2.0 / SAMPLES) * (Vref / (float)adc_res) * abs(scalei) * 1.85 / 1.4142;

    int harmonics_idx[] = {1, 3, 5, 7};

    for (int i = 0; i < 4; i++) {
    // Xác định bin trung tâm (ví dụ 50Hz là bin 10)
    int central_bin = round((50.0 * harmonics_idx[i]) / 5.0);  // 5.0 là do 5120 / 1024
    
    // Thuật toán Peak Search: Kiểm tra bin đó và 2 bin lân cận 
    // để lấy giá trị lớn nhất (bù trừ khi tần số lệch 49.9Hz hay 50.1Hz)
    float max_mag = iReal[central_bin];
    if (iReal[central_bin - 1] > max_mag) max_mag = iReal[central_bin - 1];
    if (iReal[central_bin + 1] > max_mag) max_mag = iReal[central_bin + 1];

    float amps_rms = max_mag * calibration_factor;

    Result.harmonics[i] = (amps_rms <= 0.00) ? 0 : amps_rms;
}

    return Result;
}

// Cấu hình Wifi
const char* ssid = "ESP32_Smart_Sensor";
const char* password = "12345678";

WebServer server(80);

    doluong data ={0};
    int label_index = 1;

// 6. Hàm tạo giao diện Web

void handleRoot() {
  String html = "<!DOCTYPE html><html><head>";
  
  // -- Cấu hình Meta --
  html += "<meta charset='UTF-8'>";
  // Tự động tải lại trang sau 2 giây để cập nhật số liệu mới
  html += "<meta http-equiv='refresh' content='2'>"; 
  html += "<title>ESP32 Smart sensor</title>";

  // -- CSS (Đã sửa lại cho gọn và đúng chuẩn) --
  html += "<style>";
  // Dùng text-align: center để canh giữa toàn bộ nội dung dễ dàng hơn flexbox
  html += "body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }";
  html += "h1 { color: #007BFF; }";
  html += "p { font-size: 1.2rem; line-height: 1.6; }"; // Tăng khoảng cách dòng cho dễ đọc
  html += "</style>";
  
  html += "</head><body>";
  
  // -- Nội dung chính --
  html += "<h1>ESP32 Smart sensor</h1>";
  html += "<hr>";
  
  html += "<p>";
  // Chèn giá trị biến vào HTML bằng cách cộng chuỗi
  html += "Vrms = <b>" + String(data.Vrms, 2) + "</b> (V)<br>";
  html += "Irms = <b>" + String(data.Irms, 3) + "</b> (A)<br>";
  html += "P = <b>"    + String(data.Power, 2) + "</b> (W)<br>";
  html += "S = <b>"    + String(data.bieukien, 2) + "</b> (VA)<br>";
  html += "Cosphi = <b>" + String(data.cosphi, 3) + "</b><br>";
  html += "Dự đoán tải: <span style='color:red; font-weight:bold;'>";
html += DEVICE_NAMES[label_index];
html += "</span>";
  html += "</p>";
  
  html += "<hr>";
  html += "</body></html>";

  // Gửi code HTML về trình duyệt
  server.send(200, "text/html", html);
}

// 7. MAIN SETUP & LOOP

void setup() {
    Serial.begin(115200);
    pinMode(cambiendong, INPUT);
    pinMode(cambienap, INPUT);
    analogReadResolution(12);
    delay(2000); // Chờ 2s để mở Serial Monitor kị
    Serial.println("\n\n=== KHOI DONG HE THONG AI ===");
    // BƯỚC 1: KIỂM TRA FILE MODEL
    verifyModelData();
    // BƯỚC 2: KHỞI TẠO TFLITE
    if (!setupTFLite()) {
        Serial.println("!!! LOI NGHIEM TRONG: He thong dung hoat dong !!!");
        while(1);
    }
    Serial.println("-> Setup hoan tat. Bat dau vong lap chinh...");

    // Kết nối Wifi AP
    WiFi.softAP(ssid, password);
    Serial.print("Dia chi IP cua Web: ");
    Serial.println(WiFi.softAPIP());
    
    // Bật server
    server.on("/", handleRoot);
    server.begin();
} // Đóng ngoặc setup()

void loop() {
    // 1. Đọc cảm biến và tính toán đặc trưng
    
    data = Calc();
    
    // 2. In thông số đo đạc ra Serial (để debug)
    Serial.printf("Vrms: %.2f V | Irms: %.2f A | P: %.2f W\n", data.Vrms, data.Irms, data.Power);
    Serial.printf("CosPhi: %.2f | S: %.2f VA\n", data.cosphi, data.bieukien);
    Serial.printf("Harmonics [1,3,5,7]: [%.3f, %.3f, %.3f, %.3f]\n", 
                  data.harmonics[0], data.harmonics[1], data.harmonics[2], data.harmonics[3]);

    // 3. Phân loại thiết bị bằng AI
    // Lưu ý: Hàm predict của bạn cần 7 tham số đầu vào để tạo ra 9 features bên trong
    label_index = predict(data.Power, data.cosphi, data.harmonics[0], data.harmonics[3]);

    // 4. Hiển thị kết quả
    if (label_index != -1) {
        Serial.println("==================================");
        Serial.printf("KET QUA DU DOAN: [ %s ]\n", DEVICE_NAMES[label_index]);
        Serial.println("==================================");
    } else {
        Serial.println("!!! LOI DU DOAN !!!");
    }

    server.handleClient();

    // 5. Delay trước khi đo mẫu tiếp theo (tùy chỉnh theo nhu cầu Real-time)
    delay(500);
}
