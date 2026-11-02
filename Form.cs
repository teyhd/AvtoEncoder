using NAudio.Wave;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json;
using System.Runtime.InteropServices;
using NAudio.CoreAudioApi;
using System.Drawing;
using Microsoft.Win32;
using System.Collections.Generic;
using System.Web;
using System.Threading;

namespace PlatonAlarm
{
    public partial class Form1 : Form
    {
        static bool InSchool = true;
        //static string AppPath = @"C:\Windows\secur\1";
        static string AppPath = AppDomain.CurrentDomain.BaseDirectory;
        //private System.Windows.Forms.Timer topMostForce;
        

        [DllImport("user32.dll", EntryPoint = "FindWindow", SetLastError = true)]
        static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
        [DllImport("user32.dll", EntryPoint = "SendMessage", SetLastError = true)]
        static extern IntPtr SendMessage(IntPtr hWnd, Int32 Msg, IntPtr wParam, IntPtr lParam);

        const int WM_COMMAND = 0x111;
        const int MIN_ALL = 419;
        const int MIN_ALL_UNDO = 416;

        [DllImport("winmm.dll")]
        static extern int waveOutGetVolume(IntPtr hwo, out uint dwVolume);
        public static Lesson Less = new Lesson();

        const int SPI_SETDESKWALLPAPER = 20;
        const int SPIF_UPDATEINIFILE = 0x01;
        const int SPIF_SENDWININICHANGE = 0x02;

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        static extern int SystemParametersInfo(int uAction, int uParam, string lpvParam, int fuWinIni);

        public enum Style : int
        {
            Tiled,
            Centered,
            Stretched
        }
        public Form1()
        {
            InitializeComponent();
            this.MaximizeBox = false;
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.TransparencyKey = Color.White;
            this.FormBorderStyle = FormBorderStyle.None;
            this.ShowInTaskbar = true;
            //this.StartPosition = FormStartPosition.Manual;
            this.TopMost = true;
            Screen screen = Screen.FromPoint(Cursor.Position); // Получаем текущий экран
            //this.StartPosition = FormStartPosition.Manual; // Устанавливаем ручное позиционирование формы
            //this.Location = new Point(100, screen.WorkingArea.Height- label1.Height); // Устанавливаем позицию формы в правом верхнем углу экрана
            Console.WriteLine((screen.WorkingArea.Width - this.Width - 110).ToString());
            Console.WriteLine( screen.WorkingArea.Width);
            TopMost = true;

           //this.Height = label1.Height + label2.Height + progressBar1.Height;
           // this.Height = screen.Bounds.Height;
           // this.Width = screen.Bounds.Width;

            this.Width = screen.WorkingArea.Width;
            this.Height = screen.WorkingArea.Height;

            progressBar1.Width = screen.WorkingArea.Width;
            if (IsWindows11())
            {
                progressBar1.Location = new Point(0, this.Height - progressBar1.Height - 45);
                label1.Location = new Point(130, this.Height - label1.Height - 7);
                label2.Location = new Point(this.Width - 450, this.Height - label2.Height);
                button1.Location = new Point(130, this.Height - label1.Height - 7);

              /*  progressBar1.Location = new Point(0, this.Height - progressBar1.Height - 10);
                label1.Location = new Point(20, this.Height - label1.Height - 30);
                label2.Location = new Point(this.Width - label2.Width - 20, this.Height - label2.Height - 30);
                button1.Location = new Point(130, this.Height - label1.Height - 7);*/

            }
            else
            {
                progressBar1.Location = new Point(0, this.Height - progressBar1.Height - 40);
                label1.Location = new Point(this.Width - label2.Width - 570, this.Height - label1.Height - 5);
                label2.Location = new Point(this.Width - 450, this.Height - label2.Height);
                button1.Location = new Point(130, this.Height - label2.Height);
            }
            if (InSchool)
            {
                progressBar1.Hide();
                button1.Hide();
            }
            progressBar1.Hide();
            Console.WriteLine(IsWindows11());
            // label1.Width = progressBar1.Width;
            //label2.Width = progressBar1.Width;

        }
        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= 0x80;       // WS_EX_TOOLWINDOW
                cp.ExStyle &= ~0x40000;   // remove WS_EX_APPWINDOW
                return cp;
            }
        }
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.ApplicationExitCall)
                e.Cancel = true;
            base.OnFormClosing(e);
        }
        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == Keys.Escape || keyData == (Keys.Alt | Keys.F4))
                return true;

            return base.ProcessCmdKey(ref msg, keyData);
        }
        public static bool IsWindows11()
        {
            var reg = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows NT\CurrentVersion");

            var currentBuildStr = (string)reg.GetValue("CurrentBuild");
            var currentBuild = int.Parse(currentBuildStr);

            return currentBuild >= 22000;
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            TopMost = true;
            InitTime();
            InitMelody();
           // AutoClosingMessageBox.Show("До конца урока осталось 10 минут!", "Внимание!", 10000);
            //Play("gary-meow meloboom.mp3");
            Setw();

            label1.ForeColor = System.Drawing.ColorTranslator.FromHtml("#1ABC9C");
            label2.ForeColor = System.Drawing.ColorTranslator.FromHtml("#1ABC9C");

            // Установить один раз
            NativeMethods.SetWindowPos(
                this.Handle,
                NativeMethods.HWND_TOPMOST,
                0, 0, 0, 0,
                NativeMethods.SWP_NOMOVE | NativeMethods.SWP_NOSIZE | NativeMethods.SWP_SHOWWINDOW);

            // Защита от сброса другими окнами
            // topMostForce = new System.Windows.Forms.Timer();
            // topMostForce.Interval = 3000;
            // topMostForce.Tick += (s, ev) =>
            // {

            //  };
            //topMostForce.Start();
 
         }

        [DllImport("user32.dll")]
        static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, int dwExtraInfo);

        public static List<string> ShMus = new List<string>();
        public static List<string> LongMus = new List<string>();
        
        void InitMelody()
        {
            try
            {
                string Path = $"{AppPath}\\1";
                foreach (string fileName in Directory.GetFiles(Path))
                {

                    using (var audioFile = new AudioFileReader(fileName))
                    {
                        // Console.WriteLine(audioFile.Length);
                        string[] temp = fileName.Split('\\');
                        if (audioFile.TotalTime.TotalSeconds < 8)
                        {
                            Console.WriteLine(temp[temp.Length - 1]);
                            Console.WriteLine(audioFile.TotalTime);
                            ShMus.Add(temp[temp.Length - 1]);
                            listBox1.Items.Add(temp[temp.Length - 1]);
                        }
                        else
                        {
                            LongMus.Add(temp[temp.Length - 1]);
                            Console.WriteLine(temp[temp.Length - 1]);
                            Console.WriteLine(audioFile.TotalTime);
                            listBox2.Items.Add(temp[temp.Length - 1]);
                        }

                    };
                }
            } catch (Exception e)
            {
                Console.Write(e.ToString(), "Shape processing failed.");
            }
          

        }
        Boolean flagmus = true;
        async void Play(string Path = "alarm.mp3",float Newval = 0.9f)
        {
            if (InSchool)
            {
                await Task.Delay(2);
                timer1.Enabled = true;
                return;
            }
            flagmus = false;
            Path = $"{AppPath}\\1\\{Path}";
            try
            {
                var enumerator = new MMDeviceEnumerator();
                var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
                keybd_event((byte)Keys.VolumeUp, 0, 0, 0); // increase volume
                float volumeLevel = device.AudioEndpointVolume.MasterVolumeLevelScalar;
                device.AudioEndpointVolume.MasterVolumeLevelScalar = 0.9f;
                
                using (var audioFile = new AudioFileReader(Path))
                using (var outputDevice = new WaveOutEvent())
                {
                    outputDevice.Stop();
                    outputDevice.Init(audioFile);
                    outputDevice.Volume = 0.9f;
                    outputDevice.Play();
                    await Task.Delay(5);
                    flagmus = true;
                    while (outputDevice.PlaybackState == PlaybackState.Playing)
                    {
                        if (flagmus==false) outputDevice.Stop();
                        //Console.WriteLine(outputDevice.GetPosition());
                        await Task.Delay(1);
                    }
                    
                }
                device.AudioEndpointVolume.MasterVolumeLevelScalar = volumeLevel;
                timer1.Enabled = true;
            }
            catch (Exception e)
            {
                timer1.Enabled = true;
                //LogError(e, "Shape processing failed.");
                //throw;
            }
        }

        private void progressBar1_Click(object sender, EventArgs e)
        {
            // timer1.Enabled = true;
            //PlayMelody();
            ShortSong();

        }

        static void PlayMelody()
        {
            int[,] melody = new int[,]
            {
            { 659, 200 }, // E
            { 659, 200 }, // E
            { 659, 200 }, // E
            { 523, 200 }, // C
            { 659, 200 }, // E
            { 783, 400 }, // G
            { 392, 400 }, // G
            };

            // Воспроизведение мелодии
            for (int i = 0; i < melody.GetLength(0); i++)
            {
                Console.Beep(melody[i, 0], melody[i, 1]+150);
             //   Thread.Sleep(melody[i, 1]);
                Console.WriteLine(i);
            }
        }
        public static void RereadTime()
        {
            timer1.Enabled = false;
            Less.ReloadTime();
            InitTime();
        }
        private static void InitTime(string filePath = "0.txt")
        {
            filePath = $"{AppPath}\\{filePath}";
            Console.WriteLine(filePath);
            try
            {
                string jsonContent = File.ReadAllText(filePath);
                dynamic jsonObject = JsonConvert.DeserializeObject(jsonContent);
                foreach (var item in jsonObject)
                {
                   Console.WriteLine((string)item.start);
                   Less.AddTime((string)item.start, (string)item.stop);
                }
                timer1.Enabled = true;
            }
            catch (Exception e)
            {
                Console.Write(e.ToString(), "Shape processing failed.");

                //throw;
            }

        }
        int CountShort = 0;
        int CountLong = 0;
        private void timer1_Tick(object sender, EventArgs e)
        {
           // TopMost = true;
            try
            {
                label1.Text = Less.GetText()[0];
                label2.Text = Less.GetText()[1];
                progressBar1.Text = Less.GetText()[1];
                progressBar1.Value = 100 - int.Parse(Less.GetText()[2]);
               // Console.WriteLine(Less.GetText()[1].Split(':')[1]);
                if (Less.GetText()[1].Split(':')[1] == "20")
                {
                    timer1.Enabled = false;
                    Less.ReloadTime();
                    InitTime();
                }
                if (Less.GetText()[1] == "00:10:00" && Less.GetText()[3] == "t" && !InSchool)
                {
                    //PlayMelody();
                    timer1.Enabled = false;
                    AutoClosingMessageBox.Show("До конца урока осталось 10 минут!", "Внимание!", 3000);
                    //Play("ay-ya-ya meloboom.mp3");

                    label1.ForeColor = System.Drawing.ColorTranslator.FromHtml("#4A90E2");
                    label2.ForeColor = System.Drawing.ColorTranslator.FromHtml("#4A90E2");
                    //timer1.Enabled = false;
                    timer1.Enabled = true;
                }
                if (Less.GetText()[1] == "00:05:00" && Less.GetText()[3] == "t")
                {
                    timer1.Enabled = false;
                    AutoClosingMessageBox.Show("Осталось 5 минут! Сохраните свои работы!", "Внимание!", 2000);
                   // ShortSong();
                    timer1.Enabled = true;

                    label1.ForeColor = System.Drawing.ColorTranslator.FromHtml("#FFB74D");
                    label2.ForeColor = System.Drawing.ColorTranslator.FromHtml("#FFB74D");
                }
                if (Less.GetText()[1] == "00:01:00")
                {
                    timer1.Enabled = false;
                    AutoClosingMessageBox.Show("Через минуту конец урока", "Внимание! Конец урока!", 8000);
                    Longsong();

                    label1.ForeColor = System.Drawing.ColorTranslator.FromHtml("#1ABC9C");
                    label2.ForeColor = System.Drawing.ColorTranslator.FromHtml("#1ABC9C");
                    //timer1.Enabled = true;
                }
            }
            catch (Exception ex)
            {
                //LogError(e, "Shape processing failed.");
               // throw;
            }           
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            e.Cancel = true;
        }

        public class AutoClosingMessageBox
        {
            System.Threading.Timer _timeoutTimer;
            string _caption;
            AutoClosingMessageBox(string text, string caption, int timeout)
            {
                _caption = caption;
                _timeoutTimer = new System.Threading.Timer(OnTimerElapsed,
                    null, timeout, System.Threading.Timeout.Infinite);
                using (_timeoutTimer)
                    System.Windows.Forms.MessageBox.Show(text, caption);
            }
            public static void Show(string text, string caption, int timeout)
            {
                new AutoClosingMessageBox(text, caption, timeout);
            }
            void OnTimerElapsed(object state)
            {
                IntPtr mbWnd = FindWindow("#32770", _caption); // lpClassName is #32770 for MessageBox
                if (mbWnd != IntPtr.Zero)
                    SendMessage(mbWnd, WM_CLOSE, IntPtr.Zero, IntPtr.Zero);
                _timeoutTimer.Dispose();
            }
            const int WM_CLOSE = 0x0010;
            [System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true)]
            static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
            [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
            static extern IntPtr SendMessage(IntPtr hWnd, UInt32 Msg, IntPtr wParam, IntPtr lParam);
        }

        private void label1_Click(object sender, EventArgs e)
        {
            if (CountShort >= 0) {
                CountShort--;
            }
            if (CountLong >= 0)
            {
                CountLong--;
            }
         
        }

        void Longsong()
        {
            if (CountLong >= LongMus.Count)
            {
                CountLong = 0;
                Play("Salarm.mp3");
            }
            else
            {
                Play(LongMus[CountLong],0.9f);
                CountLong++;
            }
        } 

        void ShortSong()
        {
            if (CountShort >= ShMus.Count)
            {
                CountShort = 0;
                Play("Salarm.mp3");
            }
            else
            {
                Play(ShMus[CountShort]);
                CountShort++;
            }
        }

        private void label2_Click(object sender, EventArgs e)
        {
            Longsong();
            var enumerator = new MMDeviceEnumerator();
            var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            device.AudioEndpointVolume.MasterVolumeLevelScalar = 0.9f;
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            string curItem = listBox1.SelectedItem.ToString();
            Play(curItem);
        }

        private void listBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            string curItem = listBox2.SelectedItem.ToString();
            Play(curItem,0.9f);
        }
        public int clicked = 0;
        public Random rng = new Random();

        static public int wallp = -1;
        static public int mem = 0;

        static public int GetWallpaper()
        {
            try
            {
                int max = Directory.GetFiles(@"\\fs.local\info\wallpapers").Length - 2;
                Console.WriteLine(max);
                wallp++;
                if (wallp > max)
                {
                    wallp = 0;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
                wallp = 0;
            }

            return wallp;
        }

        static public int GetWallpaperMem()
        {
            try
            {
                int max = Directory.GetFiles(@"\\fs.local\info\wallpapers\mem").Length;
                mem++;
                if (mem > max)
                {
                    mem = 0;
                }
            }
            catch (Exception e)
            {
                mem = 0;
            }
            return wallp;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            clicked++;
            //button1.Text = "Кликов: "+ clicked.ToString();
            button1.Text = clicked.ToString();
            if (IsWindows11())
            {
                button1.Location = new Point(rng.Next(10,this.Width-10), this.Height - label1.Height - 7);
            }
            else
            {
                button1.Location = new Point(rng.Next(10, this.Width - 10), this.Height - label1.Height - 7);
            }
            
            if (clicked % 10 == 0)
            {
                Setw(true);
                System.Threading.Thread.Sleep(3300);
                Setw();
            }
            if (clicked == 1005)
            {
                AutoClosingMessageBox.Show("Ты победил!!! Покажи сообщение учителю, получи 5 по информатике", "Победа!", 3000);
            }

        }
       
        public static void Setw(Boolean flag = false)
        {
            try
            {
                IntPtr lHwnd = FindWindow("Shell_TrayWnd", null);
                // свернуть
               // SendMessage(lHwnd, WM_COMMAND, (IntPtr)MIN_ALL, IntPtr.Zero);
                //  System.Threading.Thread.Sleep(2300);
                // развернуть
                // SendMessage(lHwnd, WM_COMMAND, (IntPtr)MIN_ALL_UNDO, IntPtr.Zero);

                Style style = Style.Centered;
                String uri;
                if (flag)
                { //ismem
                    uri = Path.Combine(@"\\fs.local\info\wallpapers\mem", GetWallpaperMem().ToString() + ".jpg");
                }
                else
                {
                    uri = Path.Combine(@"\\fs.local\info\wallpapers", GetWallpaper().ToString() + ".jpg");
                }

                //MessageBox.Show(uri);

                //System.IO.Stream s = new System.Net.WebClient().OpenRead(uri.ToString());

                // System.Drawing.Image img = System.Drawing.Image.FromStream(s);
                string tempPath = uri;
                // string tempPath = Path.Combine(Path.GetTempPath(), "wallpaper.bmp");
                //img.Save(tempPath, System.Drawing.Imaging.ImageFormat.Bmp);

                RegistryKey key = Registry.CurrentUser.OpenSubKey(@"Control Panel\Desktop", true);
                if (style == Style.Stretched)
                {
                    key.SetValue(@"WallpaperStyle", 2.ToString());
                    key.SetValue(@"TileWallpaper", 0.ToString());
                }

                if (style == Style.Centered)
                {
                    key.SetValue(@"WallpaperStyle", 1.ToString());
                    key.SetValue(@"TileWallpaper", 0.ToString());
                }

                if (style == Style.Tiled)
                {
                    key.SetValue(@"WallpaperStyle", 1.ToString());
                    key.SetValue(@"TileWallpaper", 1.ToString());
                }

                SystemParametersInfo(SPI_SETDESKWALLPAPER,
                    0,
                    tempPath,
                    SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE);
            } catch (Exception e)
            {
                //err
            }
            
        }

        private void timer2_Tick(object sender, EventArgs e)
        {
            NativeMethods.SetWindowPos(
            this.Handle,
            NativeMethods.HWND_TOPMOST,
            0, 0, 0, 0,
            NativeMethods.SWP_NOMOVE | NativeMethods.SWP_NOSIZE | NativeMethods.SWP_SHOWWINDOW);
        }

        private void timer3_Tick(object sender, EventArgs e)
        {

        }
    }
}
