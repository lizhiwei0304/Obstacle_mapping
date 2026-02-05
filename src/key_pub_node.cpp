/** 
 * * * @description: 
 * * * @filename: key_pub_node.cpp 
 * * * @author: wangxurui 
 * * * @date: 2025-04-13 16:27:00
**/
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <signal.h>
#include <termios.h>
#include <stdio.h>

#define KEYCODE_SPACE 0x08
#define KEYCODE_S 115

class TeleOp
{
public:
    TeleOp();
    void KeyLoop();

private:
    bool mKeyVal;
    ros::NodeHandle mNH;
    ros::Publisher mKeyPub;
    char mCLast;
};

TeleOp::TeleOp()
{
    mKeyVal = false;
    mCLast = 0;
    mKeyPub = mNH.advertise<std_msgs::Bool>("/key", 1);
}

int gFD = 0;
struct termios gCooked, gRaw;

void Quit(int sig)
{
    tcsetattr(gFD, TCSANOW, &gCooked);
    ros::shutdown();
    exit(0);
}

void TeleOp::KeyLoop()
{
    char c;
    bool dirty=false;

    // get the console in raw mode
    tcgetattr(gFD, &gCooked);
    memcpy(&gRaw, &gCooked, sizeof(struct termios));
    gRaw.c_lflag &=~ (ICANON | ECHO);
    // Setting a new line, then end of file
    gRaw.c_cc[VEOL] = 1;
    gRaw.c_cc[VEOF] = 2;
    tcsetattr(gFD, TCSANOW, &gRaw);

    puts("Reading from keyboard");
    puts("---------------------------");
    puts("Use key S to start next iteration.");

    for(;;)
    {
        // get the next event from the keyboard
        if(read(gFD, &c, 1) < 0)
        {
            perror("read():");
            exit(-1);
        }

        ROS_DEBUG("value: 0x%02X\n", c);

        if (c == KEYCODE_S && c != mCLast)
        {
            ROS_DEBUG("iterate once!");
            dirty = true;
            mKeyVal = true;
        }
        mCLast = c;
        if(dirty)
        {
            std_msgs::Bool key_val;
            key_val.data = mKeyVal;
            mKeyPub.publish(key_val);
            dirty=false;
            mKeyVal = false;
            mCLast = 0;
        }
    }

    return;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tele_op");
    TeleOp tele_op;

    signal(SIGINT,Quit);

    tele_op.KeyLoop();

    return(0);
}
