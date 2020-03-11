#include <SPI.h>
#define DIGITAL_TRIG A0

volatile byte seqA = 0;
volatile byte seqB = 0;
volatile byte cnt = 0;
//volatile byte cnt2 = 0;
volatile boolean right = false;
volatile boolean left = false;
volatile boolean button = false;
boolean backlight = true;
byte menuitem = 1;
byte page = 1;

void ISR_trigger();

void setup() {
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  
  // Enable internal pull-up resistors
  digitalWrite(A0, HIGH);
  digitalWrite(A1, HIGH);

  Serial.begin(115200);
  Serial.print(-300);  // To freeze the lower limit
  Serial.print(" ");
  Serial.print(300);  // To freeze the upper limit
  Serial.print(" ");
 
  attachInterrupt(digitalPinToInterrupt(DIGITAL_TRIG), ISR_trigger, CHANGE);
}

void loop() {

  // MAIN LOOP 
}

void ISR_trigger(){

// Else if interrupt is triggered by encoder signals
    
    // Read A and B signals
    boolean A_val = digitalRead(A0);
    boolean B_val = digitalRead(A1);
    
    // Record the A and B signals in seperate sequences
    seqA <<= 1;
    seqA |= A_val;
    
    seqB <<= 1;
    seqB |= B_val;
    
    // Mask the MSB four bits
    seqA &= 0b00001111;
    seqB &= 0b00001111;
    
    // Compare the recorded sequence with the expected sequence
    if (seqA == 0b00001001 && seqB == 0b00000011) {
//      cnt1++;
      }
     
    if (seqA == 0b00000011 && seqB == 0b00001001) {
//      cnt2++;
      }
      
    Serial.println(1);
    Serial.print(" ");

}  
