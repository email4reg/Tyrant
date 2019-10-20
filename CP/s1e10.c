#include <stdio.h>
#include <math.h>

#define INT int
#define BEGIN {
#define END }
#define IF if(
#define FI ;}
#define THEN ){
#define ELSE }else{

INT func1()
BEGIN
    INT i;
    printf("请输入您的年龄:");
    scanf("%d",&i);

    IF i < 18 THEN
        printf("您未满18周岁,不得使用这个程序!\n");
    ELSE
        printf("您已满18周岁,欢迎使用本程序,嘿嘿...\n");
    FI

    return 0;
END

int func2()
{
    char x;

    printf("请输入一个字符:\n");
    scanf("%c",&x);

    if (x >= 'A' && x <= 'Z')
    {
        x = x + 32;
    }
    else if (x >= 'a' && x <= 'z')
    {
        x = x -32;
    }
    printf("%c\n",x);
    
    return 0;
}

int func3()
{
    float heart_rate;
    int age,max_heart_rate,bpm; // 年龄、最高心率

    printf("请输入年龄:\n");
    scanf("%d",&age);

    max_heart_rate = 220 - age;
    bpm = 150;

    playSound(bpm);
    heart_rate = getHeartRate();
   
    if (heart_rate > max_heart_rate)
    {
        printf("请马上停止跑步,否则会有生命危险......\n");
    }
    else if (heart_rate > 0.85 * max_heart_rate)
    {
        printf("请放慢脚步");
        bpm -= 20;
        playSound(bpm);
    }
    else if (heart_rate < 0.75 * max_heart_rate)
    {
        printf("Come on, 请加快节奏!\n");
        bpm += 20;
        playSound(bpm);
    }
    else
    {
        playSound(bpm);
    }

    return 0;
}

int main()
{
    // func1();
    // func2();
    func3();

    return 0;
}