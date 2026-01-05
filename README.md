# Project-1
1) Web2
  File này được nhúng lên esp32 dùng để thu thập dữ liệu cho việc training MLP. Các dữ liệu được thu thập được gửi lên bằng JSON, sao khi đủ 20 mẫu sẽ gửi lên google sheets thông qua google app script. Google app script sẽ lưu tối đa 500 giá trị mới nhất, sau đó người dùng tổng hợp 500 giá trị của mỗi tải vào file sheets khác để huấn luận mô hình.

2) MLP.ipynd
   Google colab được dùng để train model và xuất file hexa sang tensorflow lite và minh họa, trực quan hóa dữ liệu

3) final_test
  File cuối cùng, nhúng lên esp32 sẽ trả về các thông số cần đo và dự đoán tải, kết quả được hiển thị trên Serial Monitor và Web bằng access point trên esp32 qua địa chỉ IP 192.168.4.1
