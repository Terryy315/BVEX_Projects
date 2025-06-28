const int backwards = 6;//IN1
const int forwards = 7;//IN2
const int buttonfor = 2;
const int buttonback = 4;

bool moved_f = false;
bool moved_b = false;
int move_f = 0;
int move_b = 0;
int duration = 0;  //set to 0 by default
int arduino_volt = 5;
int volt_max = 50;  // = 0.5 V (divide by 100), set to trip at 5 A
bool breaker_tripped = false;

//DO NOT CHANGE THIS!!!
#define ON 0
#define OFF 1
//

//use this after stop command
void reset(){
  duration = 0;
  moved_f = false;
  moved_b = false;
  move_f = 0;
  move_b = 0;
  breaker_tripped = false;
  digitalWrite(8, 1);
}

void setup() {
  Serial.begin(9600);
  pinMode(forwards,OUTPUT);
  pinMode(backwards,OUTPUT);
  pinMode(buttonfor,INPUT_PULLUP);
  pinMode(buttonback,INPUT_PULLUP);
  digitalWrite(backwards,ON);
  digitalWrite(forwards,ON);
  digitalWrite(8, 1);
}

void loop() {
//  int sensorValue = analogRead(A1);
//  float voltage = sensorValue * ( arduino_volt / 1024.00);
//  Serial.print("\n");
//  Serial.print(voltage);
  digitalWrite(forwards,OFF);
  digitalWrite(backwards,OFF);
  if (Serial.available())
    handleSerial();
  if (!breaker_tripped)
    actuator();
}

void actuator() {
  // Serial.println("ACTUATOR");
  if(move_f==OFF && moved_f==false){
//    Serial.println("locking");          //remove later
    digitalWrite(backwards,ON);
    digitalWrite(forwards,OFF);
    moved_f = true;

    //safety mechanism
    long int max = duration;
    long int t0 = millis();
    while (millis() - t0 <= max){
      if (Serial.available()){
        //stop immediately
        stop_all();
        handleSerial();
        return;
      }
//      if (millis() - t0 > 800){ //ignore initial in rush current for 800 ms
          
          if (breaker()){
            stop_all();
            long int t1 = millis() - t0;
            Serial.print(t1);
            Serial.println(",1\n");
            return;
          }
//      }
      // delay(1);
    }
    
    digitalWrite(backwards,OFF);
    moved_b = false;
    long int t1 = millis() - t0;
    Serial.print(t1);
    Serial.println(",0\n");
  }else if(move_b==OFF && !moved_b){
//    Serial.println("unlocking");      //remove later
    digitalWrite(forwards,ON);
    digitalWrite(backwards,OFF);
    moved_b = true;  


    //Safety
    long int max = duration;
    long int t0 = millis();
    while (millis() - t0 <= max){
      if (Serial.available()){
        //stop immediately
        stop_all();
        handleSerial();
        return;
      }
//      if (millis() - t0 > 800){
        if (breaker()){
        stop_all();
        long int t1 = millis() - t0;
        Serial.print(t1);
        Serial.println(",1\n");
        return;
      }
//      }
      
      // delay(1);
    }   
    digitalWrite(forwards,OFF);
    moved_f = false;
    long int t1 = millis() - t0;
    Serial.print(t1);
    Serial.println(",0\n");
  }
  // else if((move_b==OFF && move_f==OFF)&& (moved_b || moved_f)){
  //   Serial.println("NA");
  //   // digitalWrite(backwards,ON);
  //   // digitalWrite(forwards,ON);
  //   moved_b=false;
  //   moved_f=false;
  //   delay(duration);
  // }
}

void handleSerial() {
  
    String input = Serial.readStringUntil("\0");
    input.trim();

      //accept sample: 1,2000 (locking, duration 1000ms)
    int commaIndex = input.indexOf(',');
    if (commaIndex != -1) {
      char action = input.charAt(0);
      duration = input.substring(commaIndex+1).toInt();
      
      switch (action) {
      case '2':
        stop_all();
        reset();
        break;
      case '1':
//        Serial.println("1");      //remove later
        move_f = OFF;
        move_b = ON;
        break;
      case '0':
//        Serial.println("0");      //remove later
        move_f = ON;
        move_b = OFF;
        break;
      default:
        break;
    }
    
    }

}

void stop_all(){
  digitalWrite(forwards,OFF);
  digitalWrite(backwards,OFF);
  digitalWrite(8, 0);
}

//Return 0 for safe
//Return 1 for unsafe
int breaker(){
  int sensorValue = analogRead(A1);
  float voltage = sensorValue * ( arduino_volt / 1024.00);
  voltage = roundf(voltage * 100);
  int volt = voltage;
  if (volt >= volt_max){
    Serial.println("BREAKER");      //remove later
    Serial.println(volt);           //remove later
    // breaker_tripped = true;      //commented out for easier user interfacing
    return 1;
  } else {
    Serial.println(volt);           //remove later
    return 0;
  }
}
