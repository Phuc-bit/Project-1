#include <math.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>
#include <arduinoFFT.h>
#include <ArduinoJson.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

//===== CẤU HÌNH WIFI VÀ GOOGLE SHEET =====
const char* ssid = "Phuc123";
const char* password = "27092005";
const char* GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzJAqc9rI6oVQ5X5OtPEySRGudUdXD2k53VCCxGCMipPcq1tBQrSRld-AKM3QO4kyuR/exec"; 

//===== ĐỊNH NGHĨA CHÂN VÀ HẰNG SỐ =====
#define cambiendong 35
#define cambienap 34
#define scalev 1000.0  
#define scalei 8.95   
#define adc_res 4095
#define Vref 3.3
#define vdcoffset 1835
#define idcoffset 1822
#define SAMPLES 1024
#define SAMPLING_FREQUENCY 5120 // 5120 = 5 * 1024 => đo được 1/5*50 = 10 chu kỳ nhờ vậy điểm đầu và điểm cuối cùng pha

//===== CẤU TRÚC DỮ LIỆU =====
struct doluong {
    float Vrms;
    float Irms;
    float Power;
    float bieukien;
    float cosphi;
    float harmonics[4]; // Lưu hài bậc 1, 3, 5, 7
};

//===== BIẾN TOÀN CỤC =====
volatile doluong latestData; 
SemaphoreHandle_t dataMutex;
void copyToVolatile(volatile doluong& dest, const doluong& src) {
    dest.Vrms = src.Vrms;
    dest.Irms = src.Irms;
    dest.Power = src.Power;
    dest.bieukien = src.bieukien;
    dest.cosphi = src.cosphi;
    dest.harmonics[0] = src.harmonics[0];
    dest.harmonics[1] = src.harmonics[1];
    dest.harmonics[2] = src.harmonics[2];
    dest.harmonics[3] = src.harmonics[3];
}

void copyFromVolatile(doluong& dest, const volatile doluong& src) {
    dest.Vrms = src.Vrms;
    dest.Irms = src.Irms;
    dest.Power = src.Power;
    dest.bieukien = src.bieukien;
    dest.cosphi = src.cosphi;
    dest.harmonics[0] = src.harmonics[0];
    dest.harmonics[1] = src.harmonics[1];
    dest.harmonics[2] = src.harmonics[2];
    dest.harmonics[3] = src.harmonics[3];
}

float vReal[SAMPLES];
float iReal[SAMPLES];
float iImag[SAMPLES];
ArduinoFFT<float> FFT = ArduinoFFT<float>(iReal, iImag, SAMPLES, SAMPLING_FREQUENCY);

//===== HÀM PHỤ TRỢ =====
void setup_wifi() {
    Serial.print("Dang ket noi WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");
}

//===== HÀM TÍNH TOÁN VÀ FFT =====
doluong Calc() {
    unsigned long sampling_period_us = round(1000000 * (1.0 / SAMPLING_FREQUENCY));
    unsigned long microseconds = micros();
    doluong Result = {0};

    // Lấy dữ liệu liên tục và mẫu cách đều 1024 mẫu trong 0.2 giây
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

    FFT.dcRemoval(vReal, SAMPLES); 
    FFT.dcRemoval(iReal, SAMPLES); 

    //Tính toán RMS và Power (Lúc này vReal và iReal đã sạch DC)
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
    
    // Loại bỏ nhiễu trắng (Thresholding)
    Result.harmonics[i] = (amps_rms <= 0.00) ? 0 : amps_rms;
}

    return Result;
}

//===== TÁC VỤ 1: ĐO LƯỜNG (CORE 0) =====
void AdcTask(void *pvParameters) {
    for (;;) {
        doluong tempData = Calc(); // Đo RMS

        if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
            copyToVolatile(latestData, tempData);
            xSemaphoreGive(dataMutex);
        }
        
        Serial.printf("[Core 0] Measured: Vrms=%.2f, Irms=%.2f, H3=%.2f\n", tempData.Vrms, tempData.Irms, tempData.harmonics[1]);
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

//===== SETUP =====
void setup() {
    Serial.begin(115200);
    pinMode(cambiendong, INPUT);
    pinMode(cambienap, INPUT);

    setup_wifi();
    dataMutex = xSemaphoreCreateMutex();

    doluong initialData = Calc(); 
    copyToVolatile(latestData, initialData); 
    
    // Tạo Tác vụ 1 (Đọc ADC) và GHIM (PIN) nó vào CORE 0
    xTaskCreatePinnedToCore(
        AdcTask,
        "AdcTask",
        8192,  
        NULL,  
        1,     
        NULL,  
        0);    // GHIM VÀO CORE 0

     Serial.println("[Core 1] Khoi chay Tac vu Gui Google Sheet.");
}

//===== TÁC VỤ 2: GỬI DỮ LIỆU (CORE 1 - LOOP) =====

const int BATCH_SIZE = 20; // Gom 10 mẫu rồi mới gửi
doluong buffer[BATCH_SIZE];
int bufferCount = 0;

void loop() {
    // 1. Lấy dữ liệu từ Core 0 mỗi 1 giây
    doluong dataToSend;
    if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
        copyFromVolatile(dataToSend, latestData);
        xSemaphoreGive(dataMutex);
    }

    // 2. Lưu vào bộ nhớ đệm
    if (bufferCount < BATCH_SIZE) {
        buffer[bufferCount] = dataToSend;
        bufferCount++;
    }

    // 3. Khi đủ 10 mẫu thì gửi đi
    if (bufferCount >= BATCH_SIZE) {
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            http.begin(GOOGLE_SCRIPT_URL);
            http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
            http.addHeader("Content-Type", "application/json");

            // Tạo JSON Array
            JsonDocument doc;
            JsonArray array = doc.to<JsonArray>();

            for (int i = 0; i < BATCH_SIZE; i++) {
                JsonObject obj = array.add<JsonObject>();
                obj["vrms"] = buffer[i].Vrms;
                obj["irms"] = buffer[i].Irms;
                obj["power"] = buffer[i].Power;
                obj["bieukien"] = buffer[i].bieukien;
                obj["cosphi"] = buffer[i].cosphi;
                obj["h1"] = buffer[i].harmonics[0];
                obj["h3"] = buffer[i].harmonics[1];
                obj["h5"] = buffer[i].harmonics[2];
                obj["h7"] = buffer[i].harmonics[3];
                // Thêm timestamp giả định hoặc dùng thời gian thực nếu có module RTC
                obj["ts"] = millis(); 
            }

            String jsonPayload;
            serializeJson(doc, jsonPayload);

            int httpCode = http.POST(jsonPayload);
            if (httpCode > 0) {
                Serial.printf("Sent batch success: %d\n", httpCode);
                bufferCount = 0; // Reset bộ đệm sau khi gửi thành công
            }
            http.end();
        }
    }
    delay(1000); //Việc bỏ delay khiến các mẫu bị gửi giống hệt nhau
    
}