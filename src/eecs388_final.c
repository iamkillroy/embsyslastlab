#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "eecs388_lib.h"

void auto_brake(int devid)
{
    // Task1 & 2: 
    // Your code here (Use Lab 02 - Lab 04 for reference)
    // Use the directions given in the project document
    //we really ought not to care? about polling here
    //we're just waiting until we CAN read a value
    //which we're doing right here
    //we just wait until we recieve the two start things
    //also devid just means what port we're doing
    //schoonover's library does the rest for us
        if ('Y' == ser_read(devid) && 'Y' == ser_read(devid)) 
        {
            //okay lucas here
            //remmeber by reading the serial
            //we get the next bit and clean it up
            //so now we're reading agian
            uint8_t byteLow = ser_read(devid);
            uint8_t byteHigh = ser_read(devid);
            //okay rad
            //now let's convert the high and low to a (big endian? number)
            uint16_t dist = (byteHigh << 8) | byteLow;
            printf("%d", dist);
            gpio_write(RED_LED, ON);
            //distance calculation complete!
            gpio_write(RED_LED, OFF);
            gpio_write(GREEN_LED, OFF);
            gpio_write(BLUE_LED, OFF);
            if(dist > 200){
                gpio_write(GREEN_LED, ON);
            }
            else if(dist > 100){
                gpio_write(BLUE_LED, ON);
            }
            else if(dist > 60){
                gpio_write(RED_LED, ON);
            }
            else{
                gpio_write(RED_LED, ON);
                delay(100);
                gpio_write(RED_LED, OFF);
                delay(100);
            }
        }
}

int read_from_pi(int devid)
{
    // Task-3: 
    // You code goes here (Use Lab 09-option1 for reference)
    // After performing Task-2 at dnn.py code, modify this part to read angle values from Raspberry Pi.
    // Conner here writing that read function you asked for
    if(ser_read(devid) == 'A'){
        // These might need swapped based on how you set the send protocol
        highBit = ser_read(devid);
        lowBit = ser_read(devid);
    }
    // Reads the end byte if I heard you correctly last time
    ser_read();
    // Combines the two in a 16 bit value
    uint16_t combBits = (highBit << 8) | lowBit;
    // Regular return line
    return combBits;
}

void steering(int gpio, int pos)
{
    // Task-4: 
    // Your code goes here (Use Lab 05 for reference)
    // Check the project document to understand the task
}


int main()
{
    // initialize UART channels
    ser_setup(0); // uart0
    ser_setup(1); // uart1
    int pi_to_hifive = 1; //The connection with Pi uses uart 1
    int lidar_to_hifive = 0; //the lidar uses uart 0
    
    printf("\nUsing UART %d for Pi -> HiFive", pi_to_hifive);
    printf("\nUsing UART %d for Lidar -> HiFive", lidar_to_hifive);
    
    //Initializing PINs
    gpio_mode(PIN_19, OUTPUT);
    gpio_mode(RED_LED, OUTPUT);
    gpio_mode(BLUE_LED, OUTPUT);
    gpio_mode(GREEN_LED, OUTPUT);

    printf("Setup completed.\n");
    printf("Begin the main loop.\n");

    while (1) {

        auto_brake(lidar_to_hifive); // measuring distance using lidar and braking
        int angle = read_from_pi(pi_to_hifive); //getting turn direction from pi
        printf("\nangle=%d", angle) 
        int gpio = PIN_19; 
        for (int i = 0; i < 10; i++){
            // Here, we set the angle to 180 if the prediction from the DNN is a positive angle
            // and 0 if the prediction is a negative angle.
            // This is so that it is easier to see the movement of the servo.
            // You are welcome to pass the angle values directly to the steering function.
            // If the servo function is written correctly, it should still work,
            // only the movements of the servo will be more subtle
            if(angle>0){
                steering(gpio, 180);
            }
            else {
                steering(gpio,0);
            }
            
            // Uncomment the line below to see the actual angles on the servo.
            // Remember to comment out the if-else statement above!
            // steering(gpio, angle);
        }

    }
    return 0;
}
