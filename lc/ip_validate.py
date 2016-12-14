
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_IP_string', type=str, required=True, help="IP validate")
arg = parser.parse_args()

class IP_validate(object):
    def valid(self, IP):
        def is_hex(s):
            hex_digits = set("0123456789abcdefABCDEF")
            for char in s:
                if not (char in hex_digits):
                    return False
            return True
        ary = IP.split('.')

        if len(ary)==4:
            for i in xrange(len(ary)):
                if not ary[i].isdigit() or not 0<=int(ary[i]) < 256 or (ary[i][0] == '0' and len(ary[i])>1):
                    return "Neither IPv4 or IPv6"
            return "IPv4"
        ary = IP.split(':')
        if len(ary) == 8:
            for i in xrange(len(ary)):
                tmp = ary[i]
                if len(tmp) == 0 or not len(tmp) <=4 or not is_hex(tmp):
                    return "Neither IPv4 or IPv6"
            return "IPv6"
        return "Neither IPv4 or IPv6"

if __name__=="__main__":
    validator = IP_validate()
    print (validator.valid(arg.test_IP_string))

