using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;

namespace CreateDocInformation
{
    internal class Program
    {
        static void Main(string[] args)
        {
            List<char> strings = new List<char>();
            for (int i = 0; i < 8; i++)
            {
                strings.Add('a');
            }
            for (int i = 0; i < 24; i++)
            {
                strings.Add('b');
            }
            for (int i = 0; i < 9; i++)
            {
                strings.Add('c');
            }
            for (int i = 0; i < 23; i++)
            {
                strings.Add('d');
            }
            for (int i = 0; i < 249; i++)
            {
                strings.Add('e');
            }
            int countWord = 0;
            string text = "<doc><docno>1</docno>\nd d d e e e e a d e\n</doc>\n<doc>\n<docno>2</docno>\na b b b b c\n</doc>\n";
            for  (int i = 0; i < 999; i++)
            {
                if (i< strings.Count)
                {
                    int count = (19985 - countWord )/ (strings.Count - (i));
                    text += $"<doc><docno>{i+2}</docno>\n";
                    bool flag = true;
                    for (int i2 = 0; i2 < count -1; i2++)
                    {
                        countWord ++;
                        if (flag)
                        {
                            text += strings[i];
                            flag = false;
                        }
                        else
                        {
                            text += " " + strings[i];

                        }
                    }
                    text += "\n</doc>\n";
                }
                else
                {
                    text += $"<doc><docno>{i+2}</docno>\n</doc>\n";
                }
            }
            File.WriteAllText("docTestTd", text);
            Console.WriteLine(countWord + 16);

        }
    }
}
