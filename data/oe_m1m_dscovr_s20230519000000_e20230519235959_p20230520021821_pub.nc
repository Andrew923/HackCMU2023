CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230519000000_e20230519235959_p20230520021821_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-20T02:18:21.021Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-19T00:00:00.000Z   time_coverage_end         2023-05-19T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��   T          @~{�`��?�>��
@�33CJ=�`��?�=q?}p�Ag\)C��                                    Bx���  �          @\)�c�
?�\>�33@�\)C}q�c�
?�G�?}p�Af=qC��                                    Bx�L  �          @�  �c33?��
?
=q@�{Cff�c33?�Q�?�A��C�H                                    Bx��  T          @����a�?��>�ff@���C��a�?���?���A�  C�q                                    Bx� �  T          @�=q�b�\?��H>�Q�@���C�b�\?�
=?���At��C�)                                    Bx�/>  �          @���b�\?�>�ff@�=qC�=�b�\?�{?��A�ffC�\                                    Bx�=�  T          @���^�R?�(�?!G�A
=Cp��^�R?˅?�=qA�
=Cn                                    Bx�L�  �          @�G��X��@?+�AG�CL��X��?�
=?�z�A���C��                                    Bx�[0  �          @��\�Z=q@	��?#�
A��C��Z=q?�  ?�33A��C�q                                    Bx�i�  �          @���\(�@z�?\)@���C��\(�?��H?�ffA��C��                                    Bx�x|  �          @��\�g
=?�ff?z�A�
C���g
=?���?�(�A��C&f                                    Bx��"  �          @��H�dz�?���?&ffA33C�)�dz�?�(�?�ffA��RC�H                                    Bx���  
�          @��\�p  ?���?.{AG�C���p  ?��?�A��C#�                                    Bx��n  �          @��H�s�
?���?0��A(�C ��s�
?u?��A��C%�
                                    Bx��  �          @��\�z�H?fff?#�
A�C'��z�H?��?n{AT  C+Y�                                    Bx���  T          @��\��Q�>�?
=q@�C-xR��Q�>aG�?+�A  C0��                                    Bx��`  "          @��\��  ?
=q?�\@�C,aH��  >��R?+�A(�C/��                                    Bx��  �          @��\����>�?�\@�{C-�����>��?&ffA�
C0Q�                                    Bx���            @��
����?�?�RA
=qC+�R����>���?J=qA0(�C/��                                    Bx��R  �          @��
�~�R?.{?Tz�A9�C*E�~�R>��
?��\AeC/W
                                    Bx�
�  �          @�����=q>��?0��A{C-n��=q>8Q�?Q�A5�C1��                                    Bx��  �          @���33>���?:�HA Q�C/\)��33=#�
?L��A0Q�C3xR                                    Bx�(D  �          @�����\>�(�?+�A��C-����\>��?J=qA.{C1�)                                    Bx�6�  �          @����=q?
=q?0��A(�C,c���=q>�  ?W
=A:�\C0��                                    Bx�E�  �          @����?.{?5Ap�C*xR���>�p�?k�AJ�RC.�
                                    Bx�T6  �          @�z�����?(��?:�HA!C*������>�33?k�AMC/�                                    Bx�b�  �          @�(�����?�R?#�
AffC+J=����>�{?Q�A7�
C/:�                                    Bx�q�  �          @������\?   ?�@�ffC,�3���\>��?8Q�A�
C0h�                                    Bx��(  �          @�p����H?�?��@�G�C,�
���H>�z�?5A  C/�                                    Bx���  �          @�{���?+�>�ff@ÅC*���>�?(��Ap�C-��                                    Bx��t  �          @�{��(�>���>��H@�G�C.h���(�>B�\?(�A�C1T{                                    Bx��  �          @�
=��>���>�G�@�C/�f��=�?�@��C2c�                                    Bx���  �          @�����
>Ǯ>�@�z�C.�����
>B�\?�@�C1^�                                    Bx��f  �          @�z����H>\>��@��C.����H>8Q�?z�A ��C1}q                                    Bx��  �          @�����
>�Q�>��@�G�C/����
>#�
?�@�p�C1Ǯ                                    Bx��  �          @�p���(�>��
>�G�@��
C/����(�>\)?�@��HC2)                                    Bx��X  
�          @����
>�
=?��@�C..���
>B�\?(��AC1\)                                    Bx��  �          @�ff��p�>�{>���@�\)C/J=��p�>.{?   @ڏ\C1�f                                    Bx��  "          @����{>�?�\@��C-����{>��?&ffA��C0��                                    Bx�!J  T          @�
=���?�>��H@�C,����>���?&ffA��C/޸                                    Bx�/�  �          @��R��(�?(��?   @أ�C*�)��(�>�G�?5AffC-��                                    Bx�>�  
�          @�\)��z�?=p�?   @أ�C)����z�?�\?=p�A Q�C-                                      Bx�M<  
�          @�\)���
?L��?
=q@��C(�q���
?��?J=qA-�C,h�                                    Bx�[�  �          @�
=��=q?fff?#�
A33C'z���=q?(�?n{AJ�\C+��                                    Bx�j�  �          @�
=��=q?n{?!G�AQ�C'#���=q?#�
?k�AI�C+�                                    Bx�y.  �          @�
=��=q?fff?(��A�C'����=q?��?p��AN=qC+�f                                    Bx���  T          @�
=����?�  ?#�
A(�C&)����?333?uAS33C*8R                                    Bx��z  �          @�\)����?��\?!G�A	��C%������?8Q�?uAQ�C)�R                                    Bx��   T          @�\)���?u?0��A
=C&�����?#�
?�  AY�C+                                    Bx���  
�          @�����\?c�
?0��AC'�q���\?z�?uAR�\C+��                                    Bx��l  �          @����=q?u?(��AffC&Ǯ��=q?&ff?uAQG�C*��                                    Bx��  �          @�
=��=q?k�?
=A z�C'L���=q?#�
?aG�AA�C+�                                    Bx�߸  �          @�ff���\?^�R?��@�\)C(  ���\?(�?Q�A4��C+�                                     Bx��^  �          @����33?c�
?
=A   C'����33?�R?^�RA>{C+p�                                    Bx��  �          @�ff���?k�?�@��HC'5����?&ff?^�RA>�\C*�H                                    Bx��  T          @�{���?h��?
=q@�z�C'Q����?(��?Tz�A733C*�\                                    Bx�P  �          @�p���G�?aG�?�@��
C'���G�?(�?Y��A;�
C+h�                                    Bx�(�  
�          @���G�?aG�?z�@�C'����G�?�R?\(�A=�C+Y�                                    Bx�7�  �          @���G�?h��?��A�
C'T{��G�?!G�?c�
AC�
C+�                                    Bx�FB  T          @�{����?^�R?(�A�HC'�
����?
=?c�
AC�C+��                                    Bx�T�  �          @�����?p��?�RAQ�C&Ǯ����?(��?k�AK
=C*��                                    Bx�c�  �          @�{����?��\?(�A��C%������?=p�?n{AMC)��                                    Bx�r4  �          @�����?aG�?#�
Ap�C'� ����?
=?h��AJ{C+�R                                    Bx���  �          @�����?c�
?(��A��C'�\����?
=?n{AN{C+�)                                    Bx���  �          @��R��G�?h��?E�A)�C'W
��G�?�?�ffAf{C+�3                                    Bx��&  �          @�  ��=q?J=q?k�AHQ�C)���=q>��?��Ayp�C.L�                                    Bx���  �          @�Q����
?0��?Tz�A333C*u����
>�33?��\A]C/!H                                    Bx��r  �          @������\?@  ?}p�AW
=C)�)���\>�33?�Q�A�  C/(�                                    Bx��  �          @����33?�?fffAD��C,���33>W
=?�ffAd��C1�                                    Bx�ؾ  T          @�Q���=q?��?�\)Atz�C,E��=q=�?��RA�  C2\)                                    Bx��d  T          @�ff��  ?�?�z�A�(�C+�H��  =�?��A���C2E                                    Bx��
  T          @���~{?��?�=qAp��C,��~{>�?��HA���C2�                                    Bx��  �          @�ff��G�>�Q�?���Al��C.�H��G��#�
?���Az=qC4�
                                    Bx�V  �          @���y��>��H?�{A�z�C,�{�y�����
?�Q�A��\C4Q�                                    Bx�!�  T          @�p��}p�>�ff?�p�A�\)C-��}p����
?��A��\C48R                                    Bx�0�  �          @���vff?5?��A�  C)z��vff>8Q�?�ffA�G�C1Y�                                    Bx�?H  �          @�z��r�\?aG�?��A��HC&���r�\>���?�{A��C/�                                    Bx�M�  T          @��
�qG�?Q�?�A��C'���qG�>�=q?�\)A�p�C/�3                                    Bx�\�  �          @�(��l(�?��?�G�A�33C#�3�l(�>�?��
A���C,�
                                    Bx�k:  �          @����o\)?u?�  A�(�C%���o\)>�p�?�  A�G�C.Q�                                    Bx�y�  "          @����q�?Y��?��HA�33C'E�q�>�z�?�A�  C/��                                    Bx���  
�          @���r�\?\(�?���A�{C'.�r�\>���?�z�A��C/��                                    Bx��,  �          @�z��mp�?\(�?�\)A��
C&��mp�>u?�A�{C0L�                                    Bx���  "          @���n{?xQ�?�ffA�G�C%T{�n{>�p�?��Ạ�C.O\                                    Bx��x  "          @���q�>��?ٙ�A�33C0  �q녾�33?�
=A�\)C9:�                                    Bx��  "          @�z��p��>aG�?��HA¸RC0���p�׾Ǯ?�A��HC9�3                                    Bx���  �          @�z��s�
>\?�ffA�=qC.T{�s�
�B�\?�=qA�ffC6�=                                    Bx��j  �          @�z��u>�?�
=A��\C,�f�u�L��?�G�A��C4                                    Bx��  �          @�z��tz�>�ff?�  A��\C-@ �tz����?ǮA�{C5xR                                    Bx���  �          @�z��u�?   ?�p�A�(�C,� �u��#�
?���A�  C4�H                                    Bx�\  �          @���s33>�?�  A�33C,޸�s33��\)?���A�C5�                                    Bx�  �          @�33�p  >B�\?У�A�{C1��p  �\?���A��C9�
                                    Bx�)�  �          @��H�j�H<��
?��
A�p�C3�3�j�H���?�
=A\C=E                                    Bx�8N  �          @����l(�<��
?��A�(�C3���l(��!G�?��
A���C=��                                    Bx�F�  �          @�ff�p  >�33?���A��C.���p  ���R?�{A�{C8��                                    Bx�U�  �          @�p��n{>�?�=qA���C-  �n{�L��?��A�Q�C7!H                                    Bx�d@  �          @���mp�?
=q?�ffA�\)C+���mp���G�?��A���C5��                                    Bx�r�  T          @�33�j=q?�?�G�A�\)C+0��j=q��\)?���A�=qC5!H                                    Bx���  "          @�p��n{?�?�A�p�C+�
�n{��?��A�z�C5�{                                    Bx��2  "          @���k�>��?�\)A��
C,�R�k��L��?�A��
C7
                                    Bx���  �          @�{�o\)>�ff?�A��
C-8R�o\)�W
=?��A���C78R                                    Bx��~  �          @��R�r�\?
=?�(�A���C+5��r�\��?�A�Q�C4�                                    Bx��$  T          @�ff�p  ?
=?��A��
C+
=�p  �L��?��A�G�C4�{                                    Bx���  "          @�ff�p��?G�?�
=A�G�C(@ �p��>#�
?���A�G�C1��                                    Bx��p  �          @��j=q?^�R?�ffA��HC&���j=q>L��?��RA�{C0�\                                    Bx��  T          @��hQ�?�Q�?ٙ�A�\)C!�=�hQ�?��@   A��C+n                                    Bx���  �          @�p��g
=?=p�?���A߅C(\)�g
==#�
@A��C3n                                    Bx�b  �          @�ff�i��?}p�?�=qA���C$Ǯ�i��>��R@�
A�
=C/�                                    Bx�  �          @��R�g
=?��?�33A�(�C#���g
=>�33@��A���C.k�                                   Bx�"�  �          @����p  ?5?�(�A�Q�C)G��p      @A�\)C3�3                                   Bx�1T  T          @�Q��n�R>���@G�A�ffC/W
�n�R�Ǯ@   A�z�C:                                    Bx�?�  �          @�  �n�R?s33?��A�C%���n�R>�z�@ ��A��
C/�                                     Bx�N�  �          @�G��n�R?��?���A�p�C$&f�n�R>\@z�A�p�C.)                                    Bx�]F  �          @�  �hQ�?�  ?�A�Q�C ���hQ�?�@Q�A���C+\                                    Bx�k�  �          @��R�c33?���?�\A�{C�=�c33?(��@
=A�ffC)�=                                    Bx�z�  �          @�  �W�?�{?�ffA�{C
=�W�?�z�@z�B(�C!�                                    Bx��8  �          @���Y��?�=q?�p�A���C� �Y��?�33@\)A��
C!ff                                    Bx���  �          @�  �Z=q?�?޸RA�33C���Z=q?�33@  B {C!O\                                    Bx���  �          @�  �R�\?�33?�33A�z�C�q�R�\?�z�@�B33C �
                                    Bx��*  T          @�  �P��@G�?�=qA�{C@ �P��?��@��B	\)C\)                                    Bx���  T          @����O\)?��H@   A�ffCٚ�O\)?�Q�@"�\B(�C�                                    Bx��v  T          @�Q��Mp�@G�?�A�
=C�f�Mp�?�G�@�RB�
C��                                    Bx��  �          @����N�R@?���A�p�C)�N�R?�{@(�BG�C0�                                    Bx���  �          @�Q��O\)@�?�A��CQ��O\)?�\)@��B�C(�                                    Bx��h  T          @�\)�N�R@�?ٙ�A�ffC���N�R?�Q�@�
B��C��                                    Bx�  T          @�\)�L(�@
=?�(�A��RC�{�L(�?޸R@
�HA�(�Ck�                                    Bx��  �          @�\)�L(�@z�?��
A��C�f�L(�?�Q�@p�A�(�C\                                    Bx�*Z  �          @����N{@�?��RA���C��N{?�@��A�ffC�f                                    Bx�9   T          @�=q�O\)@\)?���A���Cff�O\)?�33@�A�p�C�=                                    Bx�G�  T          @�=q�N�R@�R?�Q�A�{C� �N�R?�\)@
�HA��
C�                                    Bx�VL  
�          @���N{@   ?��A�z�C5��N{?�33@Q�A���CaH                                    Bx�d�  �          @����N{@ ��?���A��C��N{?�Q�@z�A��C��                                    Bx�s�  �          @����N�R@p�?��A�p�C���N�R?�33@33A���C��                                    Bx��>  T          @����Mp�@��?�A��RC���Mp�?���@��AC�R                                    Bx���  T          @���S�
@��?�{A��RC)�S�
?�=q@�
A��C�                                    Bx���  �          @��H�Tz�@=q?�z�A�33C  �Tz�?�=q@
=A�
=C0�                                    Bx��0  "          @�=q�QG�@%�?��At(�C���QG�@�?�z�A�{C��                                    Bx���  �          @�=q�P��@#33?�(�A���C��P��@G�?�(�A�CE                                    Bx��|  T          @����Mp�@p�?�A���Cu��Mp�?�\)@��A�p�C��                                    Bx��"  
�          @�  �J�H@�H?�(�A�\)C�
�J�H?���@
�HA�G�C
                                    Bx���  
�          @����J�H@#33?���A�\)C&f�J�H?��R@z�A��C��                                    Bx��n  T          @�33�I��@,(�?�=qA�(�C}q�I��@�@�A�p�C                                    Bx�  �          @��\�Dz�@0  ?�=qA��C
+��Dz�@�@��A�
=C��                                    Bx��  �          @����>�R@333?��A���C�=�>�R@\)@Q�A�=qC!H                                    Bx�#`  �          @����<��@333?��
A�  C� �<��@  @
=A�
=C�q                                    Bx�2  T          @����A�@-p�?�=qA��RC
33�A�@	��@�A�  C��                                    Bx�@�  �          @��\�Dz�@,(�?�Q�A�p�C
�=�Dz�@@{A���C�q                                    Bx�OR  �          @����Fff@(Q�?�{A��
C���Fff@�
@�A��CO\                                    Bx�]�  �          @����>�R@0  ?�ffA�ffC	aH�>�R@(�@
=A���C�3                                    Bx�l�  �          @�33�G
=@(��?\A�p�C���G
=@G�@�A�(�C�q                                    Bx�{D  �          @�z��I��@'
=?�=qA�G�Cp��I��?�(�@z�B 
=C��                                    Bx���  T          @���@  @-p�?�{A�(�C	��@  @�
@��B��C��                                    Bx���  �          @��
�5�@@��?�
=A�z�C33�5�@=q@33A�Q�C��                                    Bx��6  �          @��H�@  @8��?�Q�A~�HC��@  @�@�\A�=qC�R                                    Bx���  �          @�=q�A�@5?�Q�A\)C�H�A�@z�@G�A���C��                                    Bx�Ă  T          @���/\)@Tz�?8Q�A��C���/\)@:�H?�(�A�ffC)                                    Bx��(  �          @�z��'
=@[�?Tz�A.�\B��3�'
=@@  ?�{Aȏ\C�                                    Bx���  �          @�33�5�@AG�?�  A��HC.�5�@{@Q�A�ffC
�
                                    Bx��t  T          @�(��@��@AG�?s33AH��C���@��@%�?�A�p�Cu�                                    Bx��  �          @���(��@Vff?�@��C B��(��@@��?˅A���CO\                                    Bx��  "          @����ff@p  =#�
?   B�ff�ff@c�
?���A��B�\                                    Bx�f  �          @�(����@~{��z��uB䙚���@w�?s33AIG�B�                                    Bx�+  T          @�=q��\@^�R����Q�B��
��\@S�
?�ffAiG�B�aH                                    Bx�9�  �          @�Q��P  @�?��HA��HC^��P  ?�G�@
=A��C�                                     Bx�HX  T          @���Fff@&ff?��\A��C
=�Fff@�@ ��A��C&f                                    Bx�V�  T          @�p��HQ�@33?�p�A���C���HQ�?�p�@
=A�ffC                                      Bx�e�  T          @�{�AG�@'
=?��\A��RC
�AG�@ff@ ��A���C&f                                    Bx�tJ  �          @�
=�A�@%?��A���Ck��A�@�\@�A�z�C��                                    Bx���  T          @�\)�7
=@:�H?��Ao\)Cs3�7
=@��?�
=A�33C}q                                    Bx���  T          @�
=�*=q@E?��Ah��C���*=q@'�?���A�z�Cc�                                    Bx��<  �          @�\)�(Q�@G�?�ffAg33C)�(Q�@*=q?���Aܣ�C��                                    Bx���  �          @�\)�%�@N{?uAQ�C ���%�@1G�?��A�ffC�f                                    Bx���  �          @���7
=@Fff?\(�A7�C���7
=@,(�?�G�A�ffC�q                                    Bx��.  T          @��\�A�@:�H?}p�AS
=C��A�@\)?���AǮC�
                                    Bx���  �          @���6ff@G�?Y��A6ffCY��6ff@.{?�  A�(�CL�                                    Bx��z  �          @�=q�.�R@QG�?8Q�A�C�f�.�R@9��?�A�(�CL�                                    Bx��   �          @�Q�� ��@XQ�?��A�B�L�� ��@B�\?˅A�\)C��                                    Bx��  �          @���z�@`  ?
=@�\)B�#��z�@J=q?�{A��\B���                                    Bx�l  T          @�
=�ff@]p�?\)@��B�W
�ff@HQ�?���A�
=B��3                                    Bx�$  �          @�  �(Q�@P��?+�AC �H�(Q�@:=q?�\)A���C{                                    Bx�2�  �          @�ff�p�@U?!G�A
�HB�Ǯ�p�@@  ?���A�33C^�                                    Bx�A^  �          @���ff@X��>�ff@�p�B�\)�ff@G
=?�
=A��
B�(�                                    Bx�P  �          @�p����@U�>�@��B��3���@C33?�
=A���C ٚ                                    Bx�^�  T          @�p��(�@W
=>���@���B���(�@E?���A��\C E                                    Bx�mP  �          @�p��@Z=q>�@�=qB����@G�?��HA�z�B��)                                    Bx�{�  �          @����Q�@W�>�Q�@���B�aH�Q�@G�?��A�
=B��R                                    Bx���  �          @�(��
=@W�>L��@5�B�\�
=@J=q?�Q�A��
B��=                                    Bx��B  �          @���R@`��>��
@�(�B��f��R@P��?�=qA�G�B�                                    Bx���  �          @�z��G�@\(�>�33@�z�B��q�G�@L(�?���A��
B��
                                    Bx���  �          @��H���@XQ�?   @�B�����@E�?�(�A��B���                                    Bx��4  �          @��H�33@b�\    �#�
B�{�33@XQ�?��
AiB�L�                                    Bx���  �          @��H���R@u��#�
���B��῞�R@vff?   @��B���                                    Bx��  "          @�33��
=@mp���{��\)B�Ǯ��
=@i��?B�\A*=qB�                                     Bx��&  �          @�p�����@n�R�k��J�HB�(�����@hQ�?\(�A?
=B�Q�                                    Bx���  �          @�\)�
=q@g�=�\)?uB�Ǯ�
=q@\(�?�{Au�B�L�                                    Bx� r  T          @�����\@fff=#�
?�B�����\@\(�?�=qAi�B�=q                                    Bx�   T          @�  �\)@fff��\)�fffB����\)@^{?xQ�AT  B�                                    Bx� +�  �          @�����@XQ쾞�R����B�
=��@Tz�?+�AffB��f                                    Bx� :d  �          @�=q��@Vff��Q���Q�B�p���@S�
?�RAQ�B�\                                    Bx� I
  �          @�����@U��
=��
=B�(����@Tz�?��@��B�u�                                    Bx� W�  �          @�=q�
�H@XQ����B�ff�
�H@Y��>�
=@�B�\                                    Bx� fV  �          @�p���@l(�=�Q�?��B����@`��?��A~ffB�                                     Bx� t�  �          @��H��\@W�����(�B�\)��\@P��?W
=A>ffB�\                                    Bx� ��  
�          @�G��
=@N�R���H�޸RB�G��
=@O\)>�ff@��
B�.                                    Bx� �H  �          @�=q�ff@K��c�
�K�B�  �ff@S33=#�
?
=B��                                    Bx� ��  �          @��׿�@Z�H�^�R�G33B����@aG�>�?�z�B�.                                    Bx� ��  �          @������@fff�G��0  B陚����@j�H>��@i��B��                                    Bx� �:  T          @�����H@aG��G��/�B�W
���H@e>u@Tz�B�B�                                    Bx� ��  �          @�=q�
=@W��aG��F�RB�(��
=@^�R=���?���B�                                     Bx� ۆ  �          @����@S33����m�B��q�@]p���\)���B�W
                                    Bx� �,  �          @�G���@QG��p���Xz�B�W
��@Y��    =uB�=q                                    Bx� ��  �          @�  �;�@�H��33����Ck��;�@)����(���Q�C	�f                                    Bx�!x  �          @|���:�H@�
���\��\)C�f�:�H@$z�z����C
�)                                    Bx�!  �          @~�R�9��@zῴz����RCW
�9��@'��8Q��&{C	�f                                    Bx�!$�  �          @�Q��Dz�@   �˅����C���Dz�@��z�H�b{Ck�                                    Bx�!3j  "          @�G��HQ�@��p����C5��HQ�@�H�W
=�@��CG�                                    Bx�!B  �          @}p��+�@(Q쿚�H����C���+�@7
=��G����C&f                                    Bx�!P�  �          @}p���@?\)�}p��h��C aH��@I������
=B��                                    Bx�!_\  T          @\)�"�\@=p��Q��>{C�\�"�\@E�<��
>��RC��                                    Bx�!n  �          @����(�@N�R�\(��G�
B�aH�(�@U=L��?333B��=                                    Bx�!|�  �          @�G��(�@C�
�n{�T��C ���(�@L�ͽu�Q�B��                                    Bx�!�N  �          @����*�H@>{�8Q��'�C��*�H@:=q?!G�AG�C�=                                    Bx�!��  �          @�G���@K���������B��
��@J�H>�@�33B�
=                                    Bx�!��  �          @�Q���@J=q��{���B����@H��?�@�(�B��{                                    Bx�!�@  T          @\)���@J�H�u�fffB�\���@Dz�?L��A8��B��
                                    Bx�!��  �          @~{�33@N{��=q�z=qB�
=�33@J�H?��Az�B���                                    Bx�!Ԍ  �          @}p��\)@A녾�(���CxR�\)@A�>���@��Cs3                                    Bx�!�2  T          @~{��@A�>8Q�@-p�C ����@7�?}p�Ak�
C.                                    Bx�!��  �          @}p��(�@G�?n{AZffB�L��(�@0��?�(�AУ�C �                                    Bx�" ~  �          @xQ��!G�@6ff��R�Q�C� �!G�@:=q>#�
@��C��                                    Bx�"$  �          @w���R@8Q콣�
��G�C���R@2�\?.{A%�C�{                                    Bx�"�  T          @x���+�@0�׾\��{C+��+�@1G�>�Q�@��C#�                                    Bx�",p  T          @{��4z�@(���
=���\C!H�4z�@*=q���H��Q�C�)                                    Bx�";  �          @|���(��@1녿h���U�Cu��(��@:�H����
=C
=                                    Bx�"I�  T          @}p��%�@9���5�&�RC���%�@?\)=�\)?�=qC�\                                    Bx�"Xb  T          @~{�#�
@:=q�=p��,z�CL��#�
@@  =L��?@  Cn                                    Bx�"g  �          @\)��R@>{�p���Yp�C����R@G
=������C �{                                    Bx�"u�  �          @}p��#33@:�H�J=q�7�C)�#33@A�    =#�
C�                                    Bx�"�T  �          @|���!G�@5�z�H�h  C���!G�@@  �W
=�C�
C�                                    Bx�"��  
�          @�  �
=@L(��   ��z�B��H�
=@Mp�>�33@�  B��=                                    Bx�"��  �          @�  ��@Mp���(����B�����@Mp�>�
=@�Q�B���                                    Bx�"�F  T          @�Q���R@P�׿.{���B��
��R@Tz�>8Q�@&ffB�Ǯ                                    Bx�"��  �          @|(���@J=q�8Q��*�HB�����@Fff?#�
A�\B��3                                    Bx�"͒  �          @~�R�
=@Mp�=�\)?�G�B��R�
=@E�?c�
AO
=B��                                    Bx�"�8  �          @�  �
=@N�R����(�B�Q��
=@HQ�?L��A8(�B�
=                                    Bx�"��  T          @|���	��@S33���Ϳ�33B�B��	��@Mp�?@  A/�B���                                    Bx�"��  �          @z�H���R@W������B�����R@R�\?:�HA,��B�Q�                                    Bx�#*  �          @x����H@<(�>��@�\Cs3��H@.�R?���A�Q�C��                                    Bx�#�  T          @w
=�Q�@>�R�Ǯ��z�C ���Q�@>�R>�p�@�33C �)                                    Bx�#%v  �          @z=q�Q�@P�׾W
=�C33B�W
�Q�@Mp�?�RA33B�33                                    Bx�#4  �          @z�H���@O\)����(�B�Q����@J�H?0��A!p�B��                                     Bx�#B�  �          @|�����@Q녽�Q쿠  B��q���@L(�?=p�A,��B��                                    Bx�#Qh  T          @\)�  @QG���{����B�(��  @P  >��H@��B�z�                                    Bx�#`  �          @~{�
�H@Q녾����33B�  �
�H@S33>�Q�@�  B�Ǯ                                    Bx�#n�  T          @|���
=q@R�\��  �j�HB���
=q@P  ?z�A=qB�W
                                    Bx�#}Z  �          @|���Q�@Tz�k��XQ�B�p��Q�@QG�?��A�B�.                                    Bx�#�   �          @~{��\@N�R=���?��B��\��\@G
=?fffAP��B��3                                    Bx�#��  T          @�  ��@QG�>�z�@�B���@Fff?��A}G�B��R                                    Bx�#�L  "          @�Q��  @S33>\@�p�B���  @Fff?�
=A���B���                                    Bx�#��  �          @�  ���@G
=?�@�G�C J=���@8Q�?��\A��\Cff                                    Bx�#Ƙ  �          @\)��@E�?+�A
=C G���@4z�?�z�A�G�C�q                                    Bx�#�>  T          @z�H��\@HQ�?�@�B�aH��\@:=q?��\A�C :�                                    Bx�#��  �          @|���(�@A�?!G�Az�C ޸�(�@1�?���A���CB�                                    Bx�#�  �          @~{� ��@?\)?.{A��C�� ��@/\)?���A���C��                                    Bx�$0  �          @�  �#33@@��?�RA�C:��#33@1G�?�=qA�\)C��                                    Bx�$�  �          @�Q����@HQ�?+�AG�B�{���@8Q�?�33A�{C�f                                    Bx�$|  �          @�  �#�
@@��?�\@��Ch��#�
@333?�(�A�
=C}q                                    Bx�$-"  �          @�Q��  @R�\>��
@���B��)�  @G
=?���A�=qB���                                    Bx�$;�  �          @\)��R@Q�>�Q�@��B�L���R@Fff?���A�Q�B�\)                                    Bx�$Jn  �          @~�R���@S33>�p�@��HB�\)���@G
=?�33A�{B�p�                                    Bx�$Y  T          @����ff@L(�?z�A=qB��)�ff@=p�?�=qA���C ��                                    Bx�$g�  �          @\)�!G�@>{?\(�AG�C=q�!G�@+�?��A�{C&f                                    Bx�$v`  �          @\)�.�R@,(�?�=qA{�Ch��.�R@
=?�
=A�Q�C+�                                    Bx�$�  �          @}p��7�@#33?z�HAf{C
^��7�@\)?ǮA�33C�3                                    Bx�$��  T          @|(��1G�@(��?aG�AO�
Cn�1G�@
=?�p�A���C��                                    Bx�$�R  �          @z�H�0  @*�H?L��A<��C���0  @��?�33A�(�C
ٚ                                    Bx�$��  
�          @xQ��(��@,(�?\(�AN{Cz��(��@=q?�(�A�z�C	�{                                    Bx�$��  �          @xQ����@8Q�?z�HAj�RC����@$z�?У�AƏ\C��                                    Bx�$�D  �          @w
=�z�@:=q?��\Atz�C ���z�@%?�Ạ�C��                                    Bx�$��  �          @x���$z�@0  ?xQ�Ah  C���$z�@��?˅A��\CL�                                    Bx�$�  �          @xQ��'
=@&ff?�Q�A���C��'
=@��?�G�A؏\C.                                    Bx�$�6  �          @z�H�0  @p�?�=qA��\C
.�0  @?�{A��C�=                                    Bx�%�  �          @y���0  @
=?�(�A�(�CL��0  ?��H?�(�A�  Cu�                                    Bx�%�  �          @xQ��A�?�
=?�
=A�C}q�A�?�ff?�A��C�H                                    Bx�%&(  �          @u�6ff@p�?���A���C0��6ff?���?��A�(�C��                                    Bx�%4�  "          @w
=�S�
?�?xQ�Aj�HC&f�S�
?�z�?�=qA�  C�                                    Bx�%Ct  �          @s33��@��?��
A��HCB���?�@�B�Cu�                                    Bx�%R  "          @u��
�H@!G�?�33A홚C��
�H@�@�HB�RC�                                    Bx�%`�  �          @w
=��\@&ff?�(�A�  CT{��\@��@G�B�\Cٚ                                    Bx�%of  �          @z�H�E?�=q?uAp  C^��E?���?���A�=qC�                                    Bx�%~  
�          @~�R�g
=?�z�=�Q�?�=qCJ=�g
=?���>�@�C�                                    Bx�%��  "          @|(��dz�?�33=#�
?&ffCE�dz�?˅>��@�  C��                                    Bx�%�X  T          @|(��aG�?޸R=��
?�33C��aG�?�
=>��@ٙ�C}q                                    Bx�%��  "          @�  �\(�@ �׾�����33C�\(�@G�>��@
=qC�=                                    Bx�%��  	�          @}p��dz�?��H<�>���C}q�dz�?�z�>���@��C�                                    Bx�%�J  �          @|���dz�?�>L��@<��C��dz�?˅?�A  C\                                    Bx�%��  �          @w
=�e?�=�?��Ck��e?�{>�(�@�z�CE                                    Bx�%�  �          @w
=�`  ?�{>�  @k�CW
�`  ?�G�?
=AC��                                    Bx�%�<  �          @w��a�?���>���@��C
=�a�?��H?(��AC�=                                    Bx�&�  T          @tz��^{?���>��R@�\)C���^{?��H?&ffAQ�C.                                    Bx�&�  �          @w
=�_\)?�\)>���@���C��_\)?\?&ffA
=Cu�                                    Bx�&.  �          @y���^{?�G�>�\)@��C)�^{?�z�?&ffAffCxR                                    Bx�&-�  �          @|(��Z=q?���>.{@ ��C33�Z=q?�\)?
=A	�CB�                                    Bx�&<z  �          @|(��XQ�@   >�\)@��HCc��XQ�?��?5A$��C                                    Bx�&K   "          @|(��W�@ ��>��R@��C0��W�?��?=p�A,z�C��                                    Bx�&Y�  �          @|���U@�>��
@��C��U?�(�?B�\A0z�C�                                    Bx�&hl  �          @|(��S33@Q�>�33@��
C#��S33@ ��?L��A:�HC��                                    Bx�&w  �          @w��O\)@>�z�@���C5��O\)?�p�?:�HA-p�C�
                                    Bx�&��  �          @}p��R�\@
=q>���@���C�q�R�\@�?Y��AE�Cc�                                    Bx�&�^  T          @{��L��@\)>�G�@�C{�L��@?fffAT(�C�
                                    Bx�&�  �          @{��L(�@\)?�\@�\)C  �L(�@�?xQ�Ad��C�                                    Bx�&��  T          @}p��J=q@33?!G�A{C�R�J=q@�?���A���C33                                    Bx�&�P  �          @{��G�@�?0��A#33C�\�G�@?�z�A�
=C8R                                    Bx�&��  �          @|(��@��@��?!G�AQ�C���@��@��?���A��C�                                    Bx�&ݜ  
�          @\)�C33@p�?8Q�A'�C)�C33@��?�(�A�ffC�                                     Bx�&�B  
�          @\)�P  @\)?
=q@�Q�Cp��P  @�?}p�Af{Cc�                                    Bx�&��  �          @}p��S�
@��>�z�@�
=C��S�
@�\?:�HA)p�C\)                                    Bx�'	�  T          @}p��QG�@��>�p�@���C��QG�@�?Q�A>ffC�\                                    Bx�'4  �          @~{�QG�@
=q?��A(�C�H�QG�?��R?��
AqG�C�                                     Bx�'&�  
�          @�Q��S�
@�?@  A,z�C\)�S�
?�?�A�\)C�
                                    Bx�'5�  �          @�  �S�
@
=?=p�A+�Cz��S�
?�?�z�A���C�                                    Bx�'D&  �          @~{�`  ?�{>8Q�@%�C��`  ?��
?��@��C�                                    Bx�'R�  "          @}p��X��?�
=?&ffA33CO\�X��?�G�?��
Ap��C�=                                    Bx�'ar  T          @~{�9��@p�?�{A��\C� �9��@�?�=qA���C�                                    Bx�'p  "          @\)�@  @(�?z�HAd��C�3�@  @(�?���A���C��                                    Bx�'~�  T          @vff�*�H@������C
p��*�H@(���G���G�C	��                                    Bx�'�d  �          @s33�
=q@"�\���
��\)Cp��
=q@5���R���B��                                    Bx�'�
  �          @u���@p���������C	\)���@#�
��(���Q�C:�                                    Bx�'��  �          @w��!G�@�����C+��!G�@=q�����CT{                                    Bx�'�V  
�          @vff�{?�G������C}q�{@(���(����C
aH                                    Bx�'��  �          @u�Q�@   ��R���C��Q�@����ff��{C�                                     Bx�'֢  T          @vff���?�\�*�H�/z�Cn���@  �G���\C}q                                    Bx�'�H  �          @y���
=?�p��#33�$�C���
=@���
�H�C	
=                                    Bx�'��  
Z          @y���*�H?�(�����\C+��*�H@Q��33���Cc�                                    Bx�(�  �          @w��.�R@
=�޸R��(�C\)�.�R@=q��ff��G�C
�f                                    Bx�(:  "          @z�H�1G�@�H�������
C
�{�1G�@(Q�Tz��C�CxR                                    Bx�(�  
�          @|��� ��@��Q���{C	�� ��@*�H���H��p�CE                                    Bx�(.�  �          @{���@z���\���CO\��@*�H�Ǯ����CQ�                                    Bx�(=,  T          @u��\?���*�H�/�C	Q���\@ff�����
C��                                    Bx�(K�  
Z          @|���ff?�p���R��
C���ff@=q��
���CJ=                                    Bx�(Zx  "          @{��?����{�{C#��@����
���RCp�                                    Bx�(i  �          @|(��ff?����$z��"�C���ff@33�
�H��
C�=                                    Bx�(w�  �          @|�����?��
�-p��-{C�����@���z��(�C�R                                    Bx�(�j  "          @{��p�?�=q�,(��,Cc��p�@33�33�G�Cٚ                                    Bx�(�  "          @z=q�{?�(��.�R�0��C0��{@���
=�(�C=q                                    Bx�(��  "          @z=q��p�?�Q��H���V�RC���p�?�G��7��=33C
k�                                    Bx�(�\  T          @xQ���R?��
�Dz��QG�C.���R?�=q�1��7
=C	h�                                    Bx�(�  "          @z=q��
?��R�E��P�C�{��
?���333�6�C
�3                                    Bx�(Ϩ  
�          @xQ����?�\)�;��D�\C5ÿ���@Q��%��'(�Cff                                    Bx�(�N  �          @x���z�?�ff�A��L(�C�)�z�?��/\)�2�RC
}q                                    Bx�(��  �          @z�H���?�G��:�H�B  C�=���?�\�)���*�\C�
                                    Bx�(��  �          @xQ��=q?����.{�2�
CJ=�=q?�ff�(���C\)                                    Bx�)
@  
Z          @xQ���?���5�<��C8R��?���%�'�
CL�                                    Bx�)�  �          @}p��"�\?���5��6��C�\�"�\?�=q�%��#z�C�                                    Bx�)'�  "          @�  �/\)?\�p���RC��/\)?�
=�	�����C�=                                    Bx�)62  T          @�=q�:�H?�=q�
=��HC�H�:�H?�(���\���HC�                                    Bx�)D�  �          @�=q�4z�?�ff�'
=� {CQ��4z�?޸R����CT{                                    Bx�)S~  �          @�Q��8Q�?����#33�Q�C\�8Q�?����
�G�C��                                    Bx�)b$  
�          @\)�7
=?��
����{C�
�7
=?ٙ����=qCQ�                                    Bx�)p�  �          @���:�H?�G��&ff�33C�\�:�H?ٙ���
C�R                                    Bx�)p  "          @����G�?�=q��H���C��G�?�p������=qC��                                    Bx�)�  �          @���c�
?�=q�����up�C�R�c�
@   �B�\�(��C�                                    Bx�)��  �          @��
�aG�?ٙ���  ��G�CE�aG�?�녿n{�Q��C�\                                    Bx�)�b  T          @��\�Mp�?�������HC\�Mp�?У׿�\��C#�                                    Bx�)�  �          @�  �fff?�{�z��=qC��fff?�
=��z���\)C��                                    Bx�)Ȯ  T          @}p��fff?�  �#�
���CW
�fff?˅��p����C(�                                    Bx�)�T  �          @~{�i��?�Q�   ���Cp��i��?�G���  �fffC��                                    Bx�)��  T          @}p��qG�?�z�L���<(�C"ٚ�qG�?�
=    �#�
C"�)                                    Bx�)��  �          @|���k�?��׾��
��z�C^��k�?����
��33C��                                    Bx�*F  T          @~{�n�R?���L�Ϳ@  C +��n�R?�=q>8Q�@#33C Q�                                    Bx�*�  �          @�Q��vff?��>�=q@z=qC$�
�vff?z�H>�G�@�z�C%��                                    Bx�* �  �          @�  �x��?\(�>��
@��\C'�)�x��?J=q>�@��HC(�=                                    Bx�*/8  �          @�  �q�?�  �u�aG�C!���q�?�  >��@	��C!                                    Bx�*=�  �          @\)�s33?��>aG�@I��C$�H�s33?}p�>Ǯ@�z�C%W
                                    Bx�*L�  �          @�  �u?8Q�>L��@>�RC)p��u?+�>��
@�{C*�                                    Bx�*[*  �          @����xQ�?}p�>B�\@,��C%���xQ�?p��>�33@�  C&T{                                    Bx�*i�  �          @~�R�x��?
=����u�C+^��x��?!G��#�
�
=C*޸                                    Bx�*xv  �          @~{�x��?&ff���
��ffC*���x��?0�׾W
=�FffC)�                                    Bx�*�  �          @����z�H?Q녽���z�C(+��z�H?Tz�<��
>�\)C(\                                    Bx�*��  �          @}p��s�
?��
=��
?�C$��s�
?�  >�  @e�C%L�                                    Bx�*�h  T          @z�H�r�\?h��>���@�=qC&k��r�\?Y��>��@�{C'\)                                    Bx�*�  �          @y���s�
?@  >��@s33C(���s�
?0��>�p�@�G�C)�f                                    Bx�*��  	�          @~{�j�H?�p�����Q�C)�j�H?�p�=�?�\C)                                    Bx�*�Z  
�          @|���c33?��R=�?�\C@ �c33?���>�Q�@��CǮ                                    Bx�*�   
�          @����fff?�p��u�Q�CY��fff?��H>aG�@I��C�                                    Bx�*��  
�          @�33�c�
?��R��\)�\)C� �c�
@ ��=#�
?   C��                                    Bx�*�L  
�          @��
�dz�@�\���R��G�CL��dz�@�
<��
>�  C                                    Bx�+
�  �          @��H�\��@z�(���p�C{�\��@	���������C!H                                    Bx�+�  �          @��H�aG�@G��������\C#��aG�@33    �uC��                                    Bx�+(>  �          @�(��g�?�(��W
=�<(�Ck��g�?�p�=���?���CQ�                                    Bx�+6�  
�          @��
�l��?�ff    �#�
C{�l��?��
>�\)@w
=CY�                                    Bx�+E�  �          @��r�\?�(�>B�\@(��C���r�\?�>�@���C=q                                    Bx�+T0  �          @��R�Vff@{�   ���
C�f�Vff@ �׽�G�����C�                                    Bx�+b�  �          @���O\)@0�׿!G��=qC���O\)@4z�L���'
=C
��                                    Bx�+q|  T          @�\)�I��@0�׿���
=C
�3�I��@4z�.{�C
�                                    Bx�+�"            @��R�P  @(�þ�(���(�C���P  @*�H���
����C�
                                    Bx�+��  X          @�ff�Q�@%���G����C� �Q�@'��#�
�\)C\)                                    Bx�+�n  T          @�p��P  @#�
������CǮ�P  @'
=�\)��{C=q                                    Bx�+�  T          @�
=�Q�@'
=�������C�=�Q�@)����\)�}p�C)                                    Bx�+��  
�          @����N�R@0�׾�ff����C���N�R@333�����C#�                                    Bx�+�`  
�          @�G��]p�@�Ϳ
=q��
=C���]p�@   �.{�{C
                                    Bx�+�  T          @����fff@33��p����Cz��fff@�����p�C&f                                    Bx�+�  
�          @���e�@�\�0���
=Cs3�e�@
=��{���C�
                                    Bx�+�R  T          @�Q��mp�?��H�5�=qC(��mp�@�\��
=���C(�                                    Bx�,�  T          @����qG�?У׿�����C�
�qG�?�Q쾏\)�xQ�C�
                                    Bx�,�  �          @�p��]p�@{��R�
=qC^��]p�@녾�z����C�{                                    Bx�,!D  �          @���aG�@ff��R�	�C8R�aG�@
�H���R��\)Ck�                                    Bx�,/�  
�          @�p��]p�@{�\)���RCL��]p�@녾k��J�HC��                                    Bx�,>�  �          @�z��Z=q@�׿(��z�C���Z=q@zᾏ\)�z=qC                                    Bx�,M6  �          @��
�^�R@
=��R�	�C���^�R@
�H���R��G�C�                                    Bx�,[�  T          @�(��O\)@zῇ��m�Ch��O\)@(��5�   C�3                                    Bx�,j�  �          @��
�:�H@(Q쿓33��Q�C	���:�H@0�׿B�\�+�C�=                                    Bx�,y(  
�          @�
=�_\)@33������\)C�)�_\)@z�    <��
CY�                                    Bx�,��  X          @�
=�K�@(Q�O\)�1�Ch��K�@-p���(���z�Cz�                                    Bx�,�t  �          @�Q��U�@p��fff�D��C���U�@$z�����Ch�                                    Bx�,�  �          @����x��?��׾\)���RC p��x��?��=L��?5C Y�                                    Bx�,��  �          @���u?��þ.{���C���u?�=q=L��?(��C��                                    Bx�,�f  �          @�33�b�\?�(���G���z�C��b�\@ �׾#�
��RCk�                                    Bx�,�  �          @�(��dz�@   ��ff��\)C�dz�@�\�.{��C=q                                    Bx�,߲  T          @�=q��  >��Ϳ��C.:���  >���\��C-+�                                    Bx�,�X  �          @������?!G��
=q��ffC+5����?333��ff��C*G�                                    Bx�,��  �          @���}p�?\(��+����C'�}p�?p�׿
=q��C&�H                                    Bx�-�            @�33�w
=?��0����HC#)�w
=?�  �����HC"                                      Bx�-J  
�          @��
�w
=?��J=q�1C#��w
=?��\�!G��Q�C!��                                    Bx�-(�  �          @�(��o\)?��}p��_\)C@ �o\)?���J=q�1�C�f                                    Bx�-7�  T          @���mp�?\�W
=�>{C�\�mp�?�{�!G��=qC��                                    Bx�-F<  T          @�z��qG�?��Ϳz�H�\��C E�qG�?�(��L���1C�                                    Bx�-T�  T          @�z��z�H?��
�=p��%p�C%8R�z�H?�\)������C$�                                    Bx�-c�  �          @����  ?\(��0����C'޸��  ?p�׿�����C&��                                    Bx�-r.  �          @�p�����?O\)�+����C(�f����?c�
�\)��Q�C'�=                                    Bx�-��  �          @�����?Tz�B�\�'�
C(^�����?k��&ff�ffC'�                                    Bx�-�z  T          @�ff�|��?�=q�W
=�9C$�q�|��?�
=�0�����C#ff                                    Bx�-�   T          @��z=q?�녿:�H�!p�C#�z=q?�(�����{C"�H                                    Bx�-��  �          @�p��z�H?�  �
=�ffC"E�z�H?��þ�
=���HC!h�                                    Bx�-�l  �          @��R�mp�?�z�   �ٙ�C���mp�?��H��  �Y��C:�                                    Bx�-�  �          @�\)�g�@
=q��p�����C.�g�@(���Q쿢�\C�{                                    Bx�-ظ  T          @�
=�j=q@33�Ǯ��(�C��j=q@�����
CG�                                    Bx�-�^  �          @��R�e@	����������C{�e@
�H���
����Cٚ                                    Bx�-�  X          @�ff�^�R@���
�k�C)�^�R@z�>�\)@r�\CL�                                    Bx�.�  �          @��[�@Q�u�aG�CB��[�@�>k�@N{CaH                                    Bx�.P  �          @��`  @녽�Q쿢�\C���`  @G�>B�\@(Q�C�                                    Bx�.!�  �          @����\��@\)�����C�R�\��@녾���\C�{                                    Bx�.0�  �          @�{�a�@�;����{C{�a�@\)�B�\�)��C��                                    Bx�.?B  �          @�\)�\(�@�����
=C^��\(�@�������z�C�                                     Bx�.M�  
�          @���XQ�@{�z���(�C�
�XQ�@!G���=q�l��CE                                    Bx�.\�  T          @��R�e@z�(�����C��e@�þ�������CE                                    Bx�.k4  �          @��b�\@
�H�u�P  Cz��b�\@(�<�>��CW
                                    Bx�.y�  �          @�
=�j�H@   ����=qCc��j�H@33��=q�n{C�\                                    Bx�.��  
�          @�\)�n{?�
=��(����
C�{�n{?�(��L���*�HC!H                                    Bx�.�&  �          @���qG�?�Q쾊=q�qG�C�f�qG�?��H��\)�s33C��                                    Bx�.��  	�          @�  �O\)@!녿����
C�3�O\)@%�����hQ�Ch�                                    Bx�.�r  
�          @�G��0  @H�ÿh���E�C8R�0  @N�R����p�Cs3                                    Bx�.�  �          @����333@Mp�����33C��333@P  �B�\��RC��                                    Bx�.Ѿ  �          @����C33@:�H������CL��C33@=p��L���/\)C޸                                    Bx�.�d  
�          @���Q�@(Q����ffC8R�Q�@*�H�����p�C�
                                    Bx�.�
            @����H��@7���Q���33C	�\�H��@8�ü��
����C	W
                                    Bx�.��  
�          @����K�@5��\����C
Y��K�@6ff�L�Ϳ:�HC

                                    Bx�/V  T          @�G��S33@.{��Q���(�C� �S33@/\)�L�Ϳ(��C@                                     Bx�/�  T          @���c�
@���\��Q�C�3�c�
@����Ϳ���C�f                                    Bx�/)�  �          @��H�dz�@(���p���(�C���dz�@{��Q쿕C\)                                    Bx�/8H  �          @��H�aG�@!G����
��
=Ck��aG�@"�\����
=C5�                                    Bx�/F�  T          @���]p�@"�\������ffCǮ�]p�@#�
�L�Ϳ(�C�=                                    Bx�/U�  �          @�G��]p�@ �׾�33��p�C
�]p�@!녽�\)�^�RC�
                                    Bx�/d:  �          @�G��`  @{��=q�k�C�\�`  @�R    <�C��                                    Bx�/r�  T          @�G��b�\@������]p�C�{�b�\@�H<#�
>\)C��                                    Bx�/��  
�          @���Z=q@\)���
��33Cٚ�Z=q@ �׽#�
�
=C��                                    Bx�/�,  T          @���\��@�;B�\�!G�C�)�\��@p�=�Q�?�33C��                                    Bx�/��  �          @����^�R@{��33��
=C�3�^�R@\)���
����Cp�                                    Bx�/�x  �          @�Q��Z=q@!G���Q����C���Z=q@"�\���
��=qCW
                                    Bx�/�  "          @�
=�W�@ �׾�33��
=CW
�W�@!녽�\)�}p�C
                                    Bx�/��  T          @�  �U�@&ff������ffC��U�@(Q����\)C��                                    Bx�/�j  T          @����Vff@'���
=���
C�f�Vff@)�������C�
                                    Bx�/�  "          @�  �X��@!G��\��C\)�X��@#33��G���  C{                                    Bx�/��  
Z          @����S�
@,(��Ǯ��  C޸�S�
@.{���Ϳ���C��                                    Bx�0\  "          @����S33@,�;�33��
=C���S33@.{�u�\(�Cp�                                    Bx�0  �          @����Vff@*�H�Ǯ��{CxR�Vff@,(���G���z�C33                                    Bx�0"�  
(          @�33�^{@&ff��z��u�C��^{@'����
�W
=C�3                                    Bx�01N  �          @���Z�H@+��������C��Z�H@,�ͽL�Ϳ&ffC�R                                    Bx�0?�  
�          @���XQ�@.{��p���=qC8R�XQ�@/\)���
��=qC��                                    Bx�0N�  �          @�33�]p�@'
=��Q���C�R�]p�@(Q콣�
����C�q                                    Bx�0]@  T          @��
�a�@"�\��p�����CT{�a�@#�
��G�����C�                                    Bx�0k�  �          @��
�^�R@'����R��=qC\�^�R@(Q�#�
��C�H                                    Bx�0z�  �          @�z��l��@G���\��
=C���l��@�
��\)�i��C{                                    Bx�0�2  X          @�(��dz�@   ����33C��dz�@!녾k��@  C��                                    Bx�0��  �          @�(��b�\@"�\�\����Cff�b�\@#�
�����C!H                                    Bx�0�~  �          @��\�`  @ �׾�p���C^��`  @"�\���ǮC�                                    Bx�0�$  �          @��\�aG�@   �u�P  C�f�aG�@ ��<#�
=��
C�                                    Bx�0��  
�          @��\�i��@����
����Cu��i��@z�>W
=@/\)C�\                                    Bx�0�p  �          @�=q�k�@G�=��
?���C@ �k�@��>��R@���Cu�                                    Bx�0�  �          @��
�S33@4z�L���(Q�C���S33@5�=��
?��Cs3                                    Bx�0�  T          @�p��;�@Q녾�����\)C�q�;�@S33���
���
C�=                                    Bx�0�b  �          @�z��<��@N�R������
=Cp��<��@P  ���
���C:�                                    Bx�1  �          @����?\)@N{�aG��;�C޸�?\)@N{=�Q�?�33C�\                                    Bx�1�  T          @�{�;�@U��.{�
�HCT{�;�@U�>��?�p�CQ�                                    Bx�1*T  T          @�p��4z�@Y���L���)��C���4z�@Y��=�?˅C�H                                    Bx�18�  �          @����:�H@Q논��
����C�3�:�H@P��>�z�@r�\C�\                                    Bx�1G�  �          @�{�G
=@J=q    =#�
C���G
=@H��>���@z�HC��                                    Bx�1VF  �          @�ff�Dz�@Mp�<#�
>#�
C�q�Dz�@L��>��
@�=qC�H                                    Bx�1d�  �          @�\)�I��@J=q���Ϳ��C��I��@J=q>L��@!�C�R                                    Bx�1s�  T          @�Q��N{@H�ý�Q쿓33C���N{@HQ�>L��@&ffC�q                                    Bx�1�8  T          @�Q��L(�@K��L�Ϳ!G�C{�L(�@J�H>�  @H��C(�                                    Bx�1��  �          @����R�\@E��#�
��C�H�R�\@E>�?���C�)                                    Bx�1��  �          @�Q��Q�@C�
�L�Ϳ�RC���Q�@C33>k�@?\)C	
=                                    Bx�1�*  �          @���U�@>�R>8Q�@�
C
.�U�@<��>�ff@��C
u�                                    Bx�1��  �          @�\)�^{@3�
=u?@  C��^{@2�\>��R@���C.                                    Bx�1�v  T          @�
=�X��@8�þ\)��ffC�\�X��@8��=�?�=qC��                                    Bx�1�  �          @�ff�L(�@Dz�>W
=@/\)C#��L(�@B�\>��H@�=qCp�                                    Bx�1��  �          @��P  @=p�>�G�@�p�C	�f�P  @:=q?333A{C
&f                                    Bx�1�h  T          @�{�G
=@H��>���@��C�G
=@Fff?(�@�{C&f                                    Bx�2  �          @�33�J�H@@A�RC�
�J�H?�33@G�A�ffC)                                    Bx�2�  �          @�(��K�@Q�?�(�A�33C:��K�@p�?�A�
=C#�                                    Bx�2#Z  �          @����C33@J�H?}p�AK
=C���C33@Dz�?�G�A�ffC��                                    Bx�22   �          @����0��@e�>B�\@B�=q�0��@c33?   @�=qB��R                                    Bx�2@�  T          @��
�-p�@n�R=�Q�?��B�{�-p�@mp�>�
=@�B�ff                                    Bx�2OL  
�          @���(�@tz��G���33B�33�(�@tz�>aG�@333B�B�                                    Bx�2]�  �          @���\)@w�=L��?#�
B�p��\)@vff>Ǯ@���B��                                    Bx�2l�  �          @����@vff��Q쿋�B�����@u>u@E�B��                                    Bx�2{>  �          @����@xQ쾏\)�`  B�=q�@x��=u?333B��                                    Bx�2��  �          @�=q��\@z�H��Q�����B���\@|(����
���RB�aH                                    Bx�2��  P          @��
���@�  �L���"�\B��H���@�Q�>\)?�Q�B��
                                    Bx�2�0  &          @��H�33@w�?
=@�\)B�aH�33@s�
?h��A:{B�8R                                    Bx�2��  �          @��\���@z=q?fffA7�B�W
���@tz�?�p�Az�RB�z�                                    Bx�2�|  �          @���ff@g�?��HA�p�B���ff@^{@   A�ffB�L�                                    Bx�2�"  T          @����ff@mp�?���A��B����ff@e?�\)A��\B�                                    Bx�2��  �          @�녿�
=@���>�Q�@���B���
=@�  ?0��A��B�k�                                    Bx�2�n  �          @�����z�@����  �J=qB�uÿ�z�@�p�=���?�(�B�ff                                    Bx�2�  T          @�
=�Q�@h��?�z�Ao�B�aH�Q�@b�\?��HA�
=B���                                    Bx�3�  �          @�33�:=q@c33?�\)A��RCc��:=q@Z�H?�33A�\)CaH                                    Bx�3`  �          @�\)�.�R@w�?�Q�AaB�p��.�R@p��?�  A��B���                                    Bx�3+  �          @�  �33@��?p��A1�B��)�33@���?��
Ar{B��f                                    Bx�39�  �          @�p���@�G�?�@���B���@�\)?c�
A"�HB�{                                    Bx�3HR  �          @��\��(�@��H?�Q�AS33Bճ3��(�@�\)?���A��\B�k�                                    Bx�3V�  �          @��\���\@��?�33ADz�B�G����\@��?�ffA��Bʽq                                    Bx�3e�  T          @�  ��(�@�33?5@�
=B��q��(�@���?�33A;�
B��)                                    Bx�3tD  �          @�\)�}p�@��H?���A`(�B�\�}p�@�
=?�\A��BɊ=                                    Bx�3��  �          @��R���H@�{?�
=A�
=B�33���H@���@�A��B�                                    Bx�3��  T          @�\)����@�z�?˅A�{B��f����@�  ?��RA�B��f                                    Bx�3�6  �          @�녾�
=@��>�(�@���B�W
��
=@�?aG�A�B�k�                                    Bx�3��  �          @���J=q@��>�{@U�B�ff�J=q@�=q?J=q@��RBÅ                                    Bx�3��  �          @�z���@��� ����{B�.��@�(���\)��B�                                    Bx�3�(  �          @�ff�ٙ�@��ÿ���=qB׽q�ٙ�@��Ϳ�33�]��B���                                    Bx�3��  �          @�{�8Q�@�zᾙ���<��B�p��8Q�@�z�>��?�z�B�p�                                    Bx�3�t  T          @ə�=�G�@�
=>L��?�{B��f=�G�@�{?+�@�{B��H                                    Bx�3�  T          @�G��8Q�@�{���1�B��)�8Q�@�Q�8Q�����B��3                                    Bx�4�  
�          @��H�E@�
=�@����  B�.�E@�{�*�H�˅B��R                                    Bx�4f  "          @�33�k�@�{�ff���B�G��k�@�33�����B���                                    Bx�4$  T          @��
?   @��R������B�\)?   @�33�ٙ��w�B���                                    Bx�42�  �          @�z�?�Q�@��������(�B���?�Q�@�G����
�c\)B��                                    Bx�4AX  �          @���@Z=q@��������B_�@Z=q@�p����H�w33Bbp�                                    Bx�4O�  �          @�{@�(�@�������=p�BG�R@�(�@�p�<�>�  BG��                                    Bx�4^�  
�          @�  @hQ�@��
�#�
���RB_��@hQ�@�33>�Q�@H��B_\)                                    Bx�4mJ  T          @�  @{@�\)=�Q�?Tz�B��\@{@��R?   @��RB�k�                                    Bx�4{�  "          @�ff@
=q@�33?�(�A[�B�@
=q@�\)?�\)A��\B��                                    Bx�4��  
�          @ə�?�33@���?uA(�B�G�?�33@�=q?���AJ�HB���                                    Bx�4�<  �          @���?s33@���=#�
>�33B���?s33@�(�>�@�z�B��{                                    Bx�4��  "          @ƸR?333@Å?+�@�p�B�?333@���?���A#�B��H                                    Bx�4��  �          @�>k�@�Q�?�z�A.{B��3>k�@��?�=qAo
=B���                                    Bx�4�.  "          @�ff>��@���?s33A�\B�.>��@�
=?���AO
=B�\                                    Bx�4��  �          @�����\@�Q�������B����\@��Ϳ�=q��\)B�ff                                    Bx�4�z  �          @�{�J=q@�p������HB�Q�J=q@�G���ff��33B��                                    Bx�4�   �          @�33>�p�@��ÿQ��p�B���>�p�@��\��ff���
B�
=                                    Bx�4��  "          @�(�?��@��׿p�����B���?��@�=q����33B��3                                    Bx�5l  �          @�33?L��@�  =�Q�?Q�B�?L��@�\)?   @�\)B��R                                    Bx�5  T          @�=q?��@��R?�G�A]p�B�Ǯ?��@��H?�z�A�\)B�G�                                    Bx�5+�  �          @��?��
@�33?ǮAk33B�
=?��
@�\)?���A�ffB���                                    Bx�5:^  T          @��?5@�  ?�G�A�33B��H?5@��@	��A�(�B��=                                    Bx�5I  �          @�33�B�\@��?�A�p�B��þB�\@�@�\A�z�B�\                                    Bx�5W�  
�          @��H?\)@���@��A�Q�B�?\)@�(�@-p�A���B�G�                                    Bx�5fP  T          @�(�?��@�=q@��A��HB�k�?��@��@#33A���B��q                                    Bx�5t�  T          @�(�?�Q�@�  @�RA�=qB�p�?�Q�@��H@%�A�{B���                                    Bx�5��  T          @���?xQ�@��
?�
=A��B�k�?xQ�@�\)@33A��B��                                    Bx�5�B  T          @�  ?���@�=q?��RA�ffB���?���@�@�A�=qB�{                                    Bx�5��  T          @Ǯ?�ff@�33?�{A��RB�?�ff@�
=@\)A�z�B�G�                                    Bx�5��  T          @�
=?k�@��H?�z�A�\)B�(�?k�@�ff@�\A��B��q                                    Bx�5�4  "          @��?��@���?�33AV�RB���?��@�p�?��
A���B�B�                                    Bx�5��  �          @��?�ff@�Q�?ǮAn�\B�W
?�ff@���?�
=A���B��                                    Bx�5ۀ  �          @�
=?˅@�ff?��A���B�Ǯ?˅@�=q@  A�(�B�\                                    Bx�5�&  T          @ƸR?5@�G�@�
A�G�B���?5@���@�A���B���                                    Bx�5��  "          @���>#�
@�
=?�
=AW�B�  >#�
@��?�A�\)B��                                    Bx�6r  �          @���>B�\@�33?���ATQ�B�33>B�\@�  ?�  A�p�B�#�                                    Bx�6  
�          @��
>�Q�@�33?�p�A�p�B��)>�Q�@�\)@ffA���B��R                                    Bx�6$�  T          @Å=�G�@�
=?�{A'�B���=�G�@�z�?�p�Aa�B�Ǯ                                    Bx�63d  "          @��
���
@�G�?   @�ffB�
=���
@�  ?aG�A�B�{                                    Bx�6B
  "          @�@�R@�z�@,(�A�B�#�@�R@��R@AG�A�ffB�                                    Bx�6P�  T          @θR?�Q�@�33@33A�33B�G�?�Q�@�{@*=qA��B��=                                    Bx�6_V  �          @˅?@  @�  ?�
=A���B��q?@  @��
@33A�  B�k�                                    Bx�6m�  �          @�p�>�G�@��?�\A�B��H>�G�@���@	��A�Q�B��3                                    Bx�6|�  �          @�?\(�@�
=@�
A���B��?\(�@��\@�HA��HB��                                    Bx�6�H  �          @��@'
=@�{?�G�AQB�u�@'
=@��\?�\)A��RB��3                                    Bx�6��  
�          @�=q?���@��?�A��RB�W
?���@�{@�A�  B�                                    Bx�6��  �          @�  ?^�R@��?�{A��B�  ?^�R@���@\)A�\)B��                                    Bx�6�:  �          @Ϯ?
=@�p�?�{A���B�
=?
=@���@�RA�p�B���                                    Bx�6��  �          @θR?5@�z�?�\)A�z�B��
?5@���@\)A�(�B��{                                    Bx�6Ԇ  "          @�
=�\@��H@��A�ffB���\@�ff@   A�(�B�G�                                    Bx�6�,  �          @�\)>B�\@�?�A��B�ff>B�\@���@�\A���B�Q�                                    Bx�6��  T          @�ff?aG�@�\)@�A���B�G�?aG�@��H@"�\A�{B��H                                    Bx�7 x  "          @Ϯ?�ff@�(�@=qA���B�\?�ff@�
=@0��AɅB�ff                                    Bx�7  �          @�?�  @���@��A��\B��{?�  @��@2�\A�\)B��                                    Bx�7�  T          @��
?�\)@���@��A�
=B���?�\)@�(�@�RA�G�B��                                    Bx�7,j  
�          @˅?�\)@��\@ffA�(�B��f?�\)@�@+�A��B���                                    Bx�7;  "          @�33?ٙ�@��
@�A�\)B�L�?ٙ�@�\)@*=qA�G�B�p�                                    Bx�7I�            @��?�  @�(�@ ��A�z�B�k�?�  @�
=@5A�=qB�z�                                    Bx�7X\  X          @�p�?Ǯ@���@%A���B�8R?Ǯ@��@:�HA؏\B�\)                                    Bx�7g  T          @�z�?\@�p�@#�
A���B��?\@�Q�@8��A֏\B��                                    Bx�7u�  T          @��H?��@���@%�A�G�B�Q�?��@��@9��A�G�B��\                                    Bx�7�N  �          @\@@�z�@;�A�\B���@@��R@Mp�A�
=B�
=                                    Bx�7��  �          @ȣ�?У�@�  @!G�A��B���?У�@��H@5�A���B�\                                    Bx�7��  
�          @�G�?�=q@�33@��A��\B�ff?�=q@�ff@-p�A�  B��{                                    Bx�7�@  �          @���?�G�@�z�@�A�{B��?�G�@��@)��AǅB�W
                                    Bx�7��  
�          @�(�?˅@�\)@�\A��RB���?˅@��H@'�A�{B�8R                                    Bx�7͌  �          @ə�?�  @�z�@p�A�z�B�z�?�  @�  @!�A��B��                                    Bx�7�2  T          @�G�?��@�{@
=A��B��)?��@��@(�A�=qB�(�                                    Bx�7��  �          @�(�?�p�@�=q?�Q�A�B��H?�p�@�ff@G�A��RB�8R                                    Bx�7�~  �          @�\)?ٙ�@�
=?���A��\B��?ٙ�@�33@�A���B��=                                    Bx�8$  �          @�G�?�@�G�?�\)A�33B�{?�@�p�@p�A�(�B��                                     Bx�8�  �          @�ff?�Q�@�
=?�(�Ax(�B�ff?�Q�@��
@�
A���B��)                                    Bx�8%p  �          @�(�?У�@�\)?���A[
=B�Q�?У�@�z�?�\A�Q�B��
                                    Bx�84  �          @���?�\@��?�=qAK�B�?�\@���?��A|z�B��=                                    Bx�8B�  �          @�=q?�(�@�z�?���A]B��?�(�@���?�\A�\)B��{                                    Bx�8Qb  �          @��H?��
@�{?�Q�A4(�B�B�?��
@��
?�  Ad��B���                                    Bx�8`  �          @�Q�?�33@�=q?�A4  B�=q?�33@�  ?�p�AdQ�B�Ǯ                                    Bx�8n�  �          @�z�@=q@���?n{A=qB��)@=q@�
=?�p�A@z�B�\)                                    Bx�8}T  �          @��@(�@��?�33A3�
B��@(�@�G�?���Ab�HB���                                    Bx�8��  
�          @�
=@G�@�?�Q�A_�B�8R@G�@��H?޸RA��B���                                    Bx�8��  �          @�?�\)@�ff?�33AZ�\B��?�\)@��
?ٙ�A��B��\                                    Bx�8�F  �          @�{?޸R@�Q�?�=qAO\)B�?޸R@�?У�A�B��                                    Bx�8��  �          @��?�\@��H?�z�A2�RB���?�\@���?��HAb�RB�\)                                    Bx�8ƒ  �          @ƸR?��
@��
?�p�A�G�B�#�?��
@�Q�@�\A�  B�Ǯ                                    Bx�8�8  �          @�
=?B�\@�p�?�  A�z�B�u�?B�\@�=q@z�A�\)B�33                                    Bx�8��  T          @�Q�?�R@�\)?�33Au��B�8R?�R@�z�?�p�A��B�                                    Bx�8�  �          @�z�?=p�@���?��ALz�B�#�?=p�@�=q?���A~{B��                                    Bx�9*  �          @�=q��
=@fff@���B5\)B�8R��
=@W�@��RB@�\B��                                    Bx�9�  �          @��ÿ���@E�@�\)BM\)B�.����@5@�z�BW��B�u�                                    Bx�9v  �          @����@@���BV�CaH��@@�z�B_�C
�{                                    Bx�9-  "          @�
=�33@�R@���BTz�C�
�33@\)@�z�B]Q�Cٚ                                    Bx�9;�  T          @��R�z�@�G�@mp�B(�B=�z�@��\@|��B(ffB�8R                                    Bx�9Jh  �          @�{>���@�z�@n�RB  B�\>���@�@~�RB!Q�B��q                                    Bx�9Y  �          @\�\)@��@��B<\)Bî�\)@tz�@�(�BH�\BĨ�                                    Bx�9g�  �          @�����(�@aG�@���BA��B�Q쿼(�@Q�@��BM{B�33                                    Bx�9vZ  �          @�Q�>�p�@���@n{BG�B�W
>�p�@�ff@~{B �B���                                    Bx�9�   �          @�z�@   @��@ffA�Q�B���@   @�p�@(Q�A�33B��                                    Bx�9��  �          @Å?˅@�=q@ ��A�(�B��q?˅@�@2�\Aٙ�B��f                                    Bx�9�L  �          @���?�z�@��@{A�33B�p�?�z�@��
@ ��A��\B��3                                    Bx�9��  �          @�(�?��@�G�@)��A�G�B�� ?��@���@;�A��B���                                    Bx�9��  �          @��?�33@�  @0��A��B�#�?�33@�33@B�\A�Q�B�(�                                    Bx�9�>  �          @��?�33@�(�@5�A�33B�Q�?�33@�
=@FffA�  B�#�                                    Bx�9��  �          @�{?���@�\)@*�HA���B�  ?���@��\@<��A㙚B��f                                    Bx�9�  �          @��?�@�  @/\)Aә�B���?�@��H@@��A�RB���                                    Bx�9�0  �          @��?�@��@!�A�z�B�k�?�@�p�@3�
A�p�B�u�                                    Bx�:�  �          @�\)?��@��H@+�A���B�B�?��@�{@<(�A�  B�B�                                    Bx�:|  �          @�p�?�Q�@�=q@+�AׅB�W
?�Q�@�p�@<(�A���B�p�                                    Bx�:&"  �          @�\)?��
@�
=@p�A\B�Q�?��
@��H@.�RAٮB�z�                                    Bx�:4�  �          @�(�?���@�p�@
=A��B�33?���@�G�@(��Ạ�B�p�                                    Bx�:Cn  �          @�  ?�Q�@���@z�A��B�ff?�Q�@�z�@%Ạ�B��\                                    Bx�:R  �          @��?�\)@�ff@33A��HB�z�?�\)@��\@$z�A�(�B�                                    Bx�:`�  �          @��\?�p�@�\)@�A��HB�\)?�p�@��@��A�{B���                                    Bx�:o`  �          @�?�z�@��R@�HA�G�B���?�z�@��\@,(�A�z�B��H                                    Bx�:~  �          @�  ?��@�Q�@+�A�ffB���?��@��@<��A�  B�                                    Bx�:��  �          @�(�?�G�@�z�@$z�Aϙ�B�z�?�G�@�  @5A���B��R                                    Bx�:�R  �          @�(�?xQ�@��R?�33A��B��?xQ�@��@�A���B��=                                    Bx�:��  T          @���?k�@���?�p�A�z�B�=q?k�@��R@G�A�=qB��                                    Bx�:��  
�          @��H?���@��R?�G�A�z�B��H?���@��@33A�{B�z�                                    Bx�:�D  �          @���?\@�ff@A��RB�\)?\@�=q@'
=A�B��\                                    Bx�:��  �          @���?�z�@�  @  A��
B���?�z�@��
@!G�A�
=B�B�                                    Bx�:�  �          @�\)?\)@�=q@G
=A�33B�L�?\)@���@W�B	�\B��)                                    Bx�:�6  �          @�?^�R@�@J=qB{B�  ?^�R@�Q�@Z�HB�B�G�                                    Bx�;�  �          @�  ?��
@�{@P  B��B�� ?��
@���@`  B�RB���                                    Bx�;�  �          @���?aG�@�ff@R�\B  B��)?aG�@���@c33B�
B��                                    Bx�;(  �          @�{?��\@���@?\)A�B�?��\@��
@P  B�B�G�                                    Bx�;-�  �          @�?��@�\)@%A��HB�\?��@��H@7
=A�ffB�p�                                    Bx�;<t  �          @�{?��@���@N�RB��B���?��@�33@^{B=qB��3                                    Bx�;K  �          @�  ?���@��\@S33B\)B��?���@���@b�\B{B���                                    Bx�;Y�  T          @�  ?�
=@��\@8Q�A��\B��\?�
=@�p�@HQ�B��B���                                    Bx�;hf  �          @��
?˅@��@<��A�\)B�aH?˅@���@L��B{B�33                                    Bx�;w  �          @��?�\)@�Q�@3�
A�\)B�Ǯ?�\)@��@Dz�A���B�                                      Bx�;��  �          @��?�\)@��@9��A�B�B�?�\)@�  @J=qBz�B�k�                                    Bx�;�X  �          @��
?��@���@�A��B��H?��@�G�@A��\B�
=                                    Bx�;��  �          @�(�@(�@�z�?���A�  B���@(�@�G�@�A�(�B���                                    Bx�;��  �          @��@%�@��\?���AY��B|Q�@%�@�  ?��A�=qBz�
                                    Bx�;�J  �          @��\@��@�G�?s33A�HB���@��@�\)?�p�AC�B��\                                    Bx�;��  �          @�33?�p�@���?�ffA&=qB�
=?�p�@��R?��AT��B��R                                    Bx�;ݖ  �          @�ff?�Q�@�ff?�z�A<��B�
=?�Q�@�(�?�
=Aj=qB��=                                    Bx�;�<  �          @��\@�\@�
=?}p�A$��B��=@�\@��?�G�AP��B�                                    Bx�;��  �          @���@�@�\)?���Ab{B���@�@���?��A�
=B�{                                    Bx�<	�  �          @��
@ ��@�33?�G�AO
=Bz�@ ��@���?�G�Az{By{                                    Bx�<.  T          @��@+�@�Q�?\A��Bm33@+�@�p�?�G�A�  Bk33                                    Bx�<&�  T          @�=q@Dz�@���?s33A��B_�@Dz�@�\)?�Q�AE�B^(�                                    Bx�<5z  �          @���@C33@�
=?��\ALz�Bd�@C33@�z�?\Aup�Bb�
                                    Bx�<D   �          @�33@*=q@�\)?��A^�RBrz�@*=q@���?���A��\Bp�
                                    Bx�<R�  �          @���?�p�@�?���A���B�?�p�@��H?�A��B�z�                                    Bx�<al  �          @��@Q�@�(�?�{A8��B�  @Q�@��?�\)Ad��B~                                    Bx�<p  �          @��H@*�H@��\?G�A ��Bt�@*�H@���?��A+�
Bs�                                    Bx�<~�  �          @�{@5@�=q?}p�A z�Bm�H@5@�Q�?�  AJ�RBl��                                    Bx�<�^  �          @�z�@2�\@�  ?���A5G�Bn=q@2�\@�?�{A_�Bl�
                                    Bx�<�  �          @��=��
@��\@'�A��B��=��
@�{@7
=B(�B��
                                    Bx�<��  �          @��H>.{@��@A�B�B��\>.{@�{@P��Bz�B�aH                                    Bx�<�P  �          @��<��
@��R@Mp�Bz�B��<��
@���@\(�B!�
B��                                     Bx�<��  �          @���>�Q�@z�H@`  B&�\B�� >�Q�@n{@l��B2�B��                                    Bx�<֜  T          @�z�>�@�33@H��B��B��>�@�p�@W�B�B�                                      Bx�<�B  �          @�(�?z�@���@5�A��HB�33?z�@�z�@EB��B��R                                    Bx�<��  T          @�녿!G�@��@aG�B!
=B�z�!G�@z=q@o\)B-\)B�W
                                    Bx�=�  �          @�{�����@n�RB(�C:�R����0��@l��B&\)C=}q                                    Bx�=4  �          @�Q�����z�@-p�A��
CL0�����p�@%A�G�CM��                                    Bx�=�  �          @�  �~{�Vff@!�A�p�C\.�~{�_\)@A�
=C]J=                                    Bx�=.�  �          @�{���R�0  @G�A�Q�CU.���R�:�H@=p�A�=qCV�q                                    Bx�==&  �          @��H��\)��  @Tz�BffCHW
��\)��Q�@N{B ��CJ:�                                    Bx�=K�  �          @�(�������ff@^�RB
ffCE޸�����޸R@X��BQ�CG��                                    Bx�=Zr  �          @������H��ff@Tz�B  CHff���H��p�@N{A���CJB�                                    Bx�=i  �          @�ff��Q쿅�@�  B)��CA�R��Q쿡G�@|(�B&Q�CDz�                                    Bx�=w�  �          @�(���{�p��@�  B,
=C@����{��z�@|(�B(�CCs3                                    Bx�=�d  T          @����(��E�@�Q�B.�C>����(��}p�@}p�B,
=CA�                                     Bx�=�
  �          @�ff��\)��z�@Z=qB��CK���\)�ff@R�\B��CM�                                    Bx�=��  �          @�G���  �z�@:=qA��CN���  ��R@1�A�
=CO�{                                    Bx�=�V  �          @�����z�(��@^�RB�C<����z�Y��@\(�BQ�C>��                                    Bx�=��  �          @�33�����@p��B!  C:������:�H@n�RB33C=\)                                    Bx�=Ϣ  �          @����b�\>W
=@�
=BG��C0�{�b�\��@�
=BH(�C4p�                                    Bx�=�H  �          @�Q��C33?O\)@�p�BYC%8R�C33?��@�ffB\�\C)�R                                    Bx�=��  �          @��R�+�>�{@�\)Bq{C,�)�+�=�\)@��Br
=C2�{                                    Bx�=��  �          @�p��J�H?@  @��BS��C&���J�H?   @�33BVQ�C+�                                    Bx�>
:  �          @�
=�{?u@�  Bs33C� �{?0��@���Bw\)C$p�                                    Bx�>�  �          @�ff�]p�?��\@���BAG�C#���]p�?J=q@�33BD��C'(�                                    Bx�>'�  T          @���l��?8Q�@w
=B6��C(��l��?   @y��B8�C,L�                                    Bx�>6,  �          @�����33=�G�@_\)B!�RC2���33����@_\)B!�RC5W
                                    Bx�>D�  T          @����`��?�p�@\)B9�\C!H�`��?��R@�=qB>ffC s3                                    Bx�>Sx  T          @��u��Ǯ@hQ�B-
=C9�=�u����@fffB+Q�C<޸                                    Bx�>b  �          @��R���ÿ&ff@eB%(�C=+����ÿ\(�@b�\B"�C@                                    Bx�>p�  �          @�(���  =�\)@P��B
=C3��  ��@P��B  C5�
                                    Bx�>j  �          @����p�?��@:=qB��C&����p�?aG�@>{B�C(��                                    Bx�>�  T          @�{�hQ�?�ff@uB5�C#�
�hQ�?Q�@y��B9G�C'5�                                    Bx�>��  �          @�
=�mp���Q�@mp�B)33CL���mp���z�@fffB#G�CO:�                                    Bx�>�\  �          @���|���J=q@:�HA�(�CZ�
�|���Tz�@.�RA���C\�                                    Bx�>�  �          @�Q��333�k�@�Bl��CF��333���H@��
BgCKW
                                    Bx�>Ȩ  �          @�z��>{��=q@��\BZ=qCL\�>{����@��BT33CPB�                                    Bx�>�N  �          @�  �S33�33@��HB:{CS�f�S33��\@}p�B2�CV�{                                    Bx�>��  
�          @�Q��5��33@��BX(�CW���5���@�BO�RC[��                                    Bx�>��  �          @�{�X�ÿ��@���BY�\C=�R�X�ÿaG�@�Q�BV��CB�H                                    Bx�?@  
�          @��
�>�R?n{@���B]G�C"�3�>�R?&ff@��B`�
C'��                                    Bx�?�  �          @�=q�5�?�{@�=qB^�CE�5�?�=q@�z�Bc��C�                                    Bx�? �  �          @�{�H��?�  @��BY�\C"J=�H��?8Q�@��B]=qC'
                                    Bx�?/2  
�          @����ff>��R@��B��C+���ff�#�
@�  B�k�C4^�                                    Bx�?=�  �          @�녿�ff��G�@�p�B�(�CJ�{��ff�O\)@�(�B��{CYxR                                    Bx�?L~  �          @�33���H���@��B�G�Cc�ÿ��H���@��
Bu�Ch&f                                    Bx�?[$  �          @�{�����  @�  B|  Cm0������$z�@�33Bo��Cp�{                                    Bx�?i�  �          @�33��33��{@�z�Bv{C`c׿�33�
=q@���Bk�Cd��                                    Bx�?xp  �          @����Ϳ��@���B�.C���������@�p�B�#�C��                                    Bx�?�  �          @�
=�	���#�
@�  B��CD�
�	���xQ�@�ffB�(�CL@                                     Bx�?��  �          @��R��=�G�@�  B��qC'Ǯ���k�@�  B���CK�3                                    Bx�?�b  �          @�(����=���@��\B�� C*�ÿ����  @�=qB���CJ�                                    Bx�?�  T          @����G���@��
B���C7}q�G���@�33B���C@�                                    Bx�?��  �          @�
=�Q�?333@��HB�{Cz�Q�>�{@�(�B���CG�                                    Bx�?�T  �          @��
��?�33@�
=B��C���?O\)@���B���Cn                                    Bx�?��  �          @�p��`�׾��@��\Ba�RC;�\�`�׿O\)@�G�B_�C@�R                                    Bx�?��  �          @θR�c33��p�@��
Baz�C9��c33�8Q�@��HB_=qC?aH                                    Bx�?�F  �          @�Q��Z�H����@�ffBa��C:�H�Z�H�=p�@��B_�C@!H                                    Bx�@
�  �          @����z=q���@���BP��C7޸�z=q�
=@���BO(�C<��                                    Bx�@�  �          @���z�H���@���BK�C<�H�z�H�k�@�33BI{CA!H                                    Bx�@(8  "          @�p����׿p��@��RB<�C@}q���׿�  @�z�B8�CDaH                                    Bx�@6�  �          @�33���H�=p�@�G�BD=qC>:����H��ff@�\)BA  CBs3                                    Bx�@E�  �          @ə����ÿ��
@�p�B2�CD�3���ÿ���@��\B.�CH5�                                    Bx�@T*  T          @˅��
=���R@�Q�B(��CF� ��
=���
@���B#Q�CI�                                    Bx�@b�  T          @�33���׿xQ�@�=qB,=qC@)���׿�G�@�  B(�\CC��                                    Bx�@qv  �          @ʏ\��Q��@�(�B0ffC5����Q��(�@��B/z�C9c�                                    Bx�@�  �          @�z�������z�@�\)B&p�CE.������Q�@�(�B!p�CH^�                                    Bx�@��  �          @ҏ\��\)��  @��BCB)��\)���
@�Q�B�\CE�                                    Bx�@�h  �          @�ff��Q쿵@xQ�B��CC���Q��
=@q�B	\)CE�q                                    Bx�@�  �          @��
���
��G�@j�HBz�CA@ ���
��G�@dz�B�\CC�q                                    Bx�@��  �          @�ff��G����R@vffB
=CC����G���  @n�RBffCFQ�                                    Bx�@�Z  �          @�ff��
=��G�@y��BCD���
=��\@r�\B
  CF                                    Bx�@�   �          @�(���=q���\@��HB$33CB����=q����@��B�CF�                                    Bx�@�  �          @�
=��(���Q�@�33B=qCC����(���(�@\)B�CF��                                    Bx�@�L  �          @�Q����׿�{@z�HB�RCE����׿��@s33B��CG��                                    Bx�A�  �          @�=q��=q����@uB	�CG+���=q��@l��Bp�CI��                                    Bx�A�  T          @�G���G�����@c33A���CD���G�����@[�A��CFp�                                    Bx�A!>  
�          @�(����\��33@a�A���CF���\���@X��A�z�CI�                                    Bx�A/�  �          @�ff��p���\)@j=qA���CC����p����@b�\A���CFW
                                    Bx�A>�  �          @߮��p���\)@p  B�CC���p����@hQ�A�{CFff                                    Bx�AM0  �          @�=q��33�У�@\)B	�\CD:���33��@w�B�\CF�H                                    Bx�A[�  �          @�����
���
@�p�Bp�CC0����
����@���B	�CF                                      Bx�Aj|  �          @���G����
@���B�RCCh���G���@���B�
CFh�                                    Bx�Ay"  �          @�Q���  ��@�p�B {CGT{��  ��@���B
=CJ��                                    Bx�A��  �          @�\)��ff��@�p�B!(�CGz���ff��@���B
=CJ�q                                    Bx�A�n  �          @���������@���B�RCG\)����{@���B��CJu�                                    Bx�A�  �          @�R�����\@���B��CE�������@��B(�CH                                    Bx�A��  �          @�R��
=�33@w�B�
CG�3��
=��@l��A�\)CJ0�                                    Bx�A�`  �          @��H��ff�Q�@]p�A��RCIaH��ff���@R�\A�p�CK��                                    Bx�A�  �          @������?333@R�\A�C,����>�ff@U�A��C/u�                                    Bx�A߬  
�          @������@fff?��A1�C�����@`��?�z�A\(�Cn                                    Bx�A�R  �          @�G��P  @�z�z�H�Q�B�=q�P  @�ff�(����B���                                    Bx�A��  �          @Ǯ��@��Ϳ�33�V�HB�q��@���z�H��B�
=                                    Bx�B�  �          @�G��;�@��þL�Ϳ���B��f�;�@���>k�@z�B��f                                    Bx�BD  �          @�(��QG�@�z�(���B��H�QG�@�p��L�Ϳ�z�B�\                                    Bx�B(�  �          @\�U�@��<��
>.{B��q�U�@�G�>��@z=qB��                                    Bx�B7�  �          @�z��Fff@�{?ٙ�A��RB�\�Fff@�G�@�
A���B�                                    Bx�BF6  �          @��H�<��@\)@@��A��B����<��@o\)@S�
B(�C E                                    Bx�BT�  T          @�{�N�R@���@<(�A��HC ���N�R@r�\@P  Bz�Cp�                                    Bx�Bc�  �          @ƸR�\(�@���?��HA�33B�.�\(�@���@A���B��                                    Bx�Br(  �          @\�P  @�@33A�=qB��=�P  @�  @�HA�Q�B���                                    Bx�B��  T          @�=q�l(�@�{?�{A�33C�q�l(�@���@{A�p�C�\                                    Bx�B�t  T          @����s�
@��?�A��Cz��s�
@���@�A���C�{                                    Bx�B�  T          @�(���=q@~�R?�@�z�C
���=q@z�H?W
=@�ffCT{                                    Bx�B��  �          @��H���R@�G�>�z�@1G�C	ٚ���R@�  ?�R@�p�C
)                                    Bx�B�f  �          @У���
=@qG��9���ң�C�)��
=@�  �%����
C
8R                                    Bx�B�  �          @�33��=q@qG��:�H�љ�Cz���=q@�  �%����C
��                                    Bx�Bز  �          @�  ���R@k��0����z�Cn���R@y���(���G�C�
                                    Bx�B�X  �          @�p���p�@e�,����p�C����p�@s33�����ffCL�                                    Bx�B��  �          @������@]p��5��ɮC�
���@k��!�����C)                                    Bx�C�  �          @�
=��G�@Fff�{���C����G�@Q녿�������C33                                    Bx�CJ  T          @��
����@=q�5���
=C�����@(���'
=��  C�3                                    Bx�C!�  �          @�33����@{�����C�����@�H�
�H��ffC0�                                    Bx�C0�  T          @�����
@B�\��p����
C�f���
@L�Ϳ��H��CL�                                    Bx�C?<  �          @��
���@'
=�0���ߙ�C�{���@5�!G���
=C�3                                    Bx�CM�  �          @�����@   �'���{C=q���@-p�������HC5�                                    Bx�C\�  �          @�G���Q�@(������C����Q�@%�˅����C��                                    Bx�Ck.  �          @�(����
@%�u��RC  ���
@&ff�#�
��G�C�f                                    Bx�Cy�  �          @��H��(�@7�?���AZ�RC}q��(�@/\)?У�A��C��                                    Bx�C�z  �          @�\)��G�@2�\?ǮAs
=C���G�@(��?�A���Cc�                                    Bx�C�   �          @�=q���\@,��?��
A���C�\���\@!�@G�A�Q�C\)                                    Bx�C��  �          @�����@*=q?^�RA�C������@$z�?���A3�
CY�                                    Bx�C�l  T          @��\���@�H?E�@��C#����@?z�HA"=qC�H                                    Bx�C�  �          @�����=q@�\����0��C\��=q@�
���
�O\)C��                                    Bx�CѸ  �          @����=q?�  �
=��{C!���=q?�
=�������
C(�                                    Bx�C�^  �          @�����
?�  �*=q��ffC&L����
?��R�"�\�˅C#�\                                    Bx�C�  T          @�����G�?�ff��R�ˮC%�{��G�?\�ff��Q�C#:�                                    Bx�C��  �          @�G���=q?�  �z���C#����=q?ٙ��
�H���HC!ff                                    Bx�DP  �          @�33��=q?����33���C'k���=q?��ÿ�
=���C%p�                                    Bx�D�  �          @��H���?G��   ��p�C+c����?u��z���Q�C)h�                                    Bx�D)�  �          @��
���?��H� �����C&�����?�녿�����HC$�=                                    Bx�D8B  �          @��R��
=?�녿Ǯ�
=C"����
=?��
��33�c�
C!#�                                    Bx�DF�  �          @�Q����?�
=����G�C!�)���?�{��{����C                                       Bx�DU�  �          @��\����?�
=��
=��33C$�f����?�{������C#�                                    Bx�Dd4  �          @�Q����?�����  �r�\C$�����?˅��{�ZffC#s3                                    Bx�Dr�  �          @�����?��ÿ�33�_\)C#ٚ���?ٙ����R�EC"�)                                    Bx�D��  �          @�{��Q�?�=q���
���C&xR��Q�?��R�����z�C$ٚ                                    Bx�D�&  �          @�p�����?�
=�����ffC%0�����?�{��p���Q�C#s3                                    Bx�D��  �          @�������@����
�C33Ch�����@	����G���ffC8R                                    Bx�D�r  �          @�����33@p������-p�CO\��33@�R���
�333C&f                                    Bx�D�  �          @�Q���p�@   �#�
��  C!aH��p�@   =#�
>���C!Q�                                    Bx�Dʾ  �          @�p���p�?��H>�  @33C#����p�?�
=>�
=@xQ�C$0�                                    Bx�D�d  �          @�{���R@p�?5@ҏ\C�����R@Q�?k�A
{C�                                    Bx�D�
  �          @�\)��
=@{>��@{C���
=@�>�@�z�CG�                                    Bx�D��  �          @����
=@(�?0��@��C�q��
=@ff?n{A=qC��                                    Bx�EV  �          @�����Q�@!�?8Q�@�G�CY���Q�@(�?z�HA33C)                                    Bx�E�  �          @���  @
=�����HQ�C�)��  @Q�����C�f                                    Bx�E"�  �          @�\)��{@7
=?8Q�@ָRCB���{@1G�?�G�A�HC�                                    Bx�E1H  �          @��H��
=@C�
?333@��
C�q��
=@>�R?�G�A  Cs3                                    Bx�E?�  �          @�=q��z�@b�\�\)����Ch���z�@e��B�\�޸RC)                                    Bx�EN�  �          @������@L��>.{?�=qC&f����@J=q?   @�  Cn                                    Bx�E]:  �          @�����33?��@��A�z�C����33?�ff@�A�
=C!k�                                    Bx�Ek�  T          @�z�����?�\)@4z�A�ffC������?Ǯ@@  Bz�CaH                                    Bx�Ez�  �          @�����  @��@��A�Cc���  @   @\)A֣�C�                                    Bx�E�,  �          @�����  @��?��HA�ffC\��  @{@p�A�{CaH                                    Bx�E��  �          @�����p�@.�R?uA.�HCǮ��p�@'
=?�  AdQ�C�                                    Bx�E�x  �          @�����(�@   ������p�CxR��(�@
=q�����|(�C�q                                    Bx�E�  �          @����ff@%��{����C!H��ff@0  �����YC��                                    Bx�E��  �          @������@'
=�L����CǮ���@+���\����C�                                    Bx�E�j  �          @��\��(�@&ff�aG��z�C����(�@&ff=�\)?E�C��                                    Bx�E�  T          @�Q���
=@�
��=q�J�HC\)��
=@=q�Q��p�CE                                    Bx�E�  �          @Ǯ����@�@2�\A�Q�Cn����?�{@AG�A�=qC��                                    Bx�E�\  �          @�\��G�@�@��\BC����G�?�33@�G�B#z�C"�H                                    Bx�F  �          @�z���(�?�{@z=qB\)C#^���(�?�z�@�=qBC'�)                                    Bx�F�  �          @�
=��Q�?�ff@,(�A��C'���Q�?}p�@4z�A���C*�f                                    Bx�F*N  T          @�  ���
?��@>{A�C&�)���
?xQ�@FffA�(�C*=q                                    Bx�F8�  �          @����z�<#�
@I��B�C3�)��zᾳ33@HQ�B��C8�=                                    Bx�FG�  �          @�33��z�?���?5@�\C!
��z�?�  ?h��A�RC!��                                    Bx�FV@  �          @Å����@\)����w
=Cc�����@G������33C)                                    Bx�Fd�  �          @����  @
�H�5��=qC� ��  @�R��ff���HC��                                    Bx�Fs�  �          @\��  ?�\)��\)�+�C"
=��  ?�녽u�(�C!ٚ                                    Bx�F�2  �          @��\���?�  ���
�J�HC%����?\�\)��C$�\                                    Bx�F��  �          @�����
=?У׿+���C#c���
=?ٙ������ffC"�q                                    Bx�F�~  �          @��H��=q?�ff��
=��(�C#�R��=q?˅�u���C#T{                                    Bx�F�$  �          @��H��{?�G������Tz�C%���{?�녿���8  C$:�                                    Bx�F��  �          @�(���33?��R��Q���ffC$�H��33?�Q���
��
=C"�f                                    Bx�F�p  �          @�p���=q?��   ���\C$޸��=q?�33������C"�                                    Bx�F�  �          @�ff����?�G��ff���RC$s3����?�  ��
=��z�C!��                                    Bx�F�  �          @������
@�
��33�r=qCp����
@p���{�?
=C�                                    Bx�F�b  �          @�33���H@�
����a�CL����H@�Ϳ�  �-p�C�                                    Bx�G  �          @�G�����?�{���R�YG�C�q����?��R��  �/
=CQ�                                    Bx�G�  T          @�z�����@�\<#�
>.{C\����@�>�=q@I��C@                                     Bx�G#T  �          @�33���?��
�h���1�C�����?�׿0�����Cz�                                    Bx�G1�  �          @��
���R?�{���
���\C  ���R?��ÿ�������C�\                                    Bx�G@�  �          @���\)?����ff�ʸRC"G���\)?��Ϳ�z����C:�                                    Bx�GOF  �          @�z����
?��H��ϙ�C#�f���
?��H�����C u�                                    Bx�G]�  �          @�33�z=q?������C�q�z=q?������{CG�                                    Bx�Gl�  �          @�{��z�?�(���R�ڏ\C#����z�?��R��
��ffC .                                    Bx�G{8  �          @��H�\)?���p���Q�C���\)@
=q�������C��                                    Bx�G��  �          @��^{?c�
�Vff�,��C%���^{?���Mp��#�CO\                                    Bx�G��  �          @�(��j�H?�G��'
=��HCn�j�H@��
=��=qCxR                                    Bx�G�*  �          @�  �QG�?�p��K��=qC�QG�@Q��9����C�3                                    Bx�G��  �          @��\�Y��@ ���K��\)Cp��Y��@=q�8���
=C��                                    Bx�G�v  �          @����aG�?�{�Z�H�"��C0��aG�@�\�I���Q�C�R                                    Bx�G�  �          @�G�����?�  �mp�� �Cp�����@�R�\���z�C�                                    Bx�G��  �          @������?}p��^�R�
=C&}q���?�Q��Tz��ffC �                                    Bx�G�h  T          @�
=��{���I�����C5�
��{>����H���  C0(�                                    Bx�G�  �          @�����0���7�����C;�)��������<(����HC7Q�                                    Bx�H�  �          @�33��Q����$z���G�C9�q��Q쾀  �(Q����
C6�                                    Bx�HZ  T          @�\)��{�Q�������C;�{��{���H�{��Q�C8�                                    Bx�H+   �          @�\���ÿ�
=�p����HC>:����ÿTz��&ff��ffC;E                                    Bx�H9�  �          @ڏ\���R�����H����p�C>.���R�!G��P����Q�C:�                                    Bx�HHL  �          @������Ϳ���Z=q����C>(����Ϳz��aG����C9��                                    Bx�HV�  �          @�p���z�O\)�c33��z�C;�)��zᾣ�
�g���z�C7�                                    Bx�He�  �          @���  �^�R�Vff��C<B���  �����\(���ffC7�=                                    Bx�Ht>  �          @�Q���33�h���7���z�C;ٚ��33�   �>{��\)C8Y�                                    Bx�H��  �          @�{�ڏ\�Q��5���
C:�)�ڏ\����;�����C7z�                                    Bx�H��  T          @�  ����J=q�-p���{C;\��������333��(�C7�)                                    Bx�H�0  �          @��H��녿��%����C9#���녾L���(Q����C5�=                                    Bx�H��  �          @��
�˅��\�#�
����C8���˅����&ff��(�C5O\                                    Bx�H�|  �          @ۅ���þ�
=�.{����C7�{���ü��0  ���C4B�                                    Bx�H�"  �          @����;�(��'
=���HC7�\���ͽL���(����\)C4n                                    Bx�H��  �          @�{��=q���H�5��\)C8h���=q��\)�8Q���Q�C4�f                                    Bx�H�n  �          @��H���ÿ5�L������C:k����þu�QG����
C6&f                                    Bx�H�  �          @�����G��#�
�U�ޣ�C9Ǯ��G��\)�X������C5O\                                    Bx�I�  �          @�
=���H����W
=��z�C9�f���H�����Z�H��Q�C4��                                    Bx�I`  �          @���녿\)�U���G�C9J=��녽�\)�W���RC4�H                                    Bx�I$  �          @߮�Å�(���U��ffC:#��Å�#�
�Y�����HC5xR                                    Bx�I2�  �          @�����33�(���Z=q���C:8R��33����^{��33C5k�                                    Bx�IAR  �          @��Ǯ�8Q��Tz��޸RC:�{�Ǯ�aG��X�����
C6                                      Bx�IO�  �          @�33��녿��L���֏\C9.��녽�Q��P  ��(�C4Ǯ                                    Bx�I^�  �          @�z���녿z��O\)���HC98R��녽��
�R�\�܏\C4�                                     Bx�ImD  �          @�����녿(���c�
���C9�R��녽�G��g����HC5�                                    Bx�I{�  �          @�����þ���hQ���C8E����=�Q��j=q��C333                                    Bx�I��  �          @���ff��G��l����=qC8
=��ff>��n�R���C2Ǯ                                    Bx�I�6  �          @���������g�����C9&f���<#�
�j=q��{C3�                                    Bx�I��  �          @�\��녿��g���(�C8�����=#�
�j=q����C3�
                                    Bx�I��  �          @�\)��p��
=�h������C9�3��p����
�k���G�C433                                    Bx�I�(  �          @�{��Q쿁G��i���z�C>\)��Q��(��qG��	�C8n                                    Bx�I��  �          @�ff���H�aG��e��z�C<�����H���R�k��=qC7!H                                    Bx�I�t  �          @ָR��=q�.{�j�H���C;  ��=q�����n�R�(�C5
=                                    Bx�I�  �          @������R�#�
�g
=� z�C:ff���R�u�j=q��RC4��                                    Bx�I��  �          @ٙ���\)����hQ�� �HC9xR��\)<��j�H��C3��                                    Bx�Jf  �          @�(����Ϳ5�vff�G�C;)���ͽ��
�z�H�
��C4�
                                    Bx�J  �          @�����Ϳ+����H���C;{���ͼ#�
��z���C4�                                    Bx�J+�  �          @ٙ����
�Tz������!=qC=B����
����\)�$��C5O\                                    Bx�J:X  �          @�
=���
�J=q��Q����C<���
��Q����H�!  C5                                    Bx�JH�  �          @�33���Ϳ\(�������C=u����;B�\����z�C6&f                                    Bx�JW�  �          @����z�O\)�����#�RC=k���z�������
�'33C5.                                    Bx�JfJ  �          @����z�fff�����"�
C>s3��z�B�\���
�'  C6:�                                    Bx�Jt�  T          @�z���
=�\(����R�$�C>\)��
=�#�
��G��)  C5�3                                    Bx�J��  �          @�\)��녿G������%{C==q��녽�\)���H�(p�C4�=                                    Bx�J�<  �          @�����p��Q���(��%G�C=���p���Q����R�(��C5                                    Bx�J��  �          @�(�����\)�g��z�C?�����\�p  �
(�C9O\                                    Bx�J��  �          @�33��녿��
�~�R�=qCB=q��녿����z��G�C:�                                    Bx�J�.  �          @љ����������H�%��CC8R����
=q����-�C:�=                                    Bx�J��  �          @�Q���녿�����  �/{CA�f��녾�33���
�5Q�C8p�                                    Bx�J�z  �          @�Q���Q�Tz�������C=u���Q�\)��z��\)C5��                                    Bx�J�   �          @�������&ff�j�H�ffC:������#�
�n�R�C4#�                                    Bx�J��  �          @�G�����}p����H�=qC?G������\)��ff� =qC7:�                                    Bx�Kl  �          @�(�������
����  C?�������z���33�$(�C7O\                                    Bx�K  �          @أ������=q�����CBk�����
=���\�33C:�                                    Bx�K$�  �          @�z����aG������ffC=�����#�
��z��ffC5��                                    Bx�K3^  �          @�G���ff�G���G��C<z���ff���
������C433                                    Bx�KB  �          @��H���\�.{��\)���C;J=���\=�\)��G��G�C3E                                    Bx�KP�  �          @�p����׿Tz��|���  C<����׽��������C5W
                                    Bx�K_P  �          @�p����Ϳc�
��p��&p�C>W
���ͽ�����Q��*�\C5.                                    Bx�Km�  �          @�(���z�\(����
�1ffC>n��z����{�5=qC4O\                                    Bx�K|�  �          @�\)���H�c�
��
=�.C>�3���H���
�����3{C5�                                    Bx�K�B  �          @����H������Q��.ffCA�����H��z���(��4�RC7�{                                    Bx�K��  �          @�\)�w���(���G��3z�CEs3�w������{�<G�C:�R                                    Bx�K��  �          @����xQ쿰���w��,��CG�\�xQ�(�������7Q�C=��                                    Bx�K�4  �          @����i����ff��Q��:=qCD{�i�����R��z��A�HC8�)                                    Bx�K��  �          @���h�ÿ}p����H�=(�CC=q�h�þk���ff�D{C7�                                    Bx�KԀ  �          @��\�c�
��{�����@Q�CEG��c�
������G��H��C9W
                                    Bx�K�&  �          @����p�׿�ff���F\)CG)�p�׾�(����\�OC:u�                                    Bx�K��  �          @��
�`�׿����Q��E{CE=q�`�׾�z���z��M33C8��                                    Bx�L r  T          @�Q��Z�H��{����S��CM33�Z�H�(����ff�`��C>�                                    Bx�L  �          @����I�������  �b��CK�I���Ǯ��p��nz�C;�                                    Bx�L�  "          @�=q�Y�����R���
�Z�CG���Y�������Q��d{C8n                                    Bx�L,d  T          @���mp����
��\)�Q(�CC� �mp����
���H�Wz�C5@                                     Bx�L;
  T          @Ǯ��zῦff��\)�7�CEh���z��G���z��@\)C:�                                    Bx�LI�  
�          @�33�qG��������
�@(�CDE�qG������  �G�
C7�                                    Bx�LXV  
(          @���aG��������?\)CDn�aG���  ���G33C8)                                    Bx�Lf�  
�          @����`  ��33��33�@ffCF&f�`  ��33����I�C9��                                    Bx�Lu�  �          @�
=�~{���\��=q�:
=CBY��~{�.{��{�@�C6��                                    Bx�L�H  �          @�z���Q�n{�����>CA)��Q�L�����
�D(�C4��                                    Bx�L��  T          @�G���  ���������9��CI�\��  �333��  �E�
C=�                                    Bx�L��  �          @�\)��z�������� �CGz���z�O\)��z��+�HC=�f                                    Bx�L�:  �          @�Q���(�����e���CE����(��h���u��G�C>{                                    Bx�L��  "          @�  ��=q�������H�G�CD�3��=q��R��G��%�RC;aH                                    Bx�L͆  T          @�G���G���\)�`����CB�\��G��(���mp��p�C;!H                                    Bx�L�,  �          @��H��z῏\)�L(���z�C?8R��z����Vff���C8�                                    Bx�L��  T          @Ӆ���Ϳ�\)�(Q���p�C>� ���Ϳz��333��ffC9�)                                    Bx�L�x  �          @�=q��  �G��!����C;n��  ��=q�(Q����C6��                                    Bx�M  T          @Ϯ����z��:�H����C9Ǯ���<#�
�>�R��\)C3��                                    Bx�M�  �          @�����þ�Q��E���z�C7������>���E��G�C1h�                                    Bx�M%j  �          @�ff��ff�#�
�6ff�ә�C:s3��ff�u�:�H��G�C4��                                    Bx�M4  �          @��H���ý�Q��N�R�陚C4�����?\)�K����
C.}q                                    Bx�MB�  �          @�
=��\)�u�[���\)C4����\)?!G��XQ�����C-�q                                    Bx�MQ\  �          @����\)����Z�H���C6z���\)>�G��Y����z�C/�\                                    Bx�M`  �          @�\)��G��k��vff�z�C6O\��G�?���tz��33C.�=                                    Bx�Mn�  �          @�z����׾.{�n{�{C5�R����?
=�j�H�ffC..                                    Bx�M}N  �          @׮������N�R��33C4B����?!G��J=q��ffC-޸                                    Bx�M��  �          @����{>�  �l�����C1����{?�  �dz���33C*p�                                    Bx�M��  �          @�G�����B�\�j�H����C5�)���?��hQ�����C.�)                                    Bx�M�@  �          @�=q������  �dz���Q�C6\)����>��H�c33��ffC/Q�                                    Bx�M��  �          @�(����R���
�x����\C4Ǯ���R?:�H�tz����C-�                                    Bx�Mƌ  �          @�(���33���
����
�HC7���33?
=q��G��	��C.�                                    Bx�M�2  �          @��
���
�0���^�R��C:�3���
=L���b�\����C3}q                                    Bx�M��  �          @ڏ\���H���`�����C9����H>k��b�\����C1�=                                    Bx�M�~  �          @�ff��녾���qG����C6�����?
=q�o\)�z�C.�3                                    Bx�N$  �          @�\)��ff>�����  �=qC1  ��ff?���vff�(�C(�3                                    Bx�N�  �          @�{���H�������33C5�����H?8Q�������C,��                                    Bx�Np  �          @߮���׾aG���  �{C6Q�����?0����ff�  C,�                                    Bx�N-  �          @߮�����p���=q��C7�3���?�\����
=C.�\                                    Bx�N;�  �          @����
=��  ������RC6����
=?#�
������C-J=                                    Bx�NJb  �          @�p���G����
���
��C40���G�?^�R������
C+�                                    Bx�NY  �          @�p���(�<#�
���\�\)C3���(�?p����\)���C*�                                    Bx�Ng�  �          @θR����    ����p�C3������?aG��}p��
=C*
=                                    Bx�NvT  �          @�����p�>#�
�������C28R��p�?����Q���C(k�                                    Bx�N��  �          @�33�����  ��=q��C6� ���?&ff�������C,ٚ                                    Bx�N��  �          @������Ϳ!G���G��(�C;W
����>�=q���\�  C0�{                                    Bx�N�F  �          @Ϯ��z�u�qG���C>����z�����x���\)C5#�                                    Bx�N��  �          @�=q����=�\)�A���z�C3W
����?:�H�<(���p�C,�=                                    Bx�N��  �          @�����?�\)��(���{C)������?��
��
=�l��C&                                    Bx�N�8  �          @���\)?u�ff��\)C+0���\)?�z��=q��  C'B�                                    Bx�N��  �          @����z�?O\)�=p����C,+���z�?����-p���ffC&(�                                    Bx�N�  �          @أ���z�?\(��Mp����HC+����z�?Ǯ�<(����C%(�                                    Bx�N�*  �          @�p���{?���S33��\)C)&f��{?�ff�>{�ӮC"ff                                    Bx�O�  �          @���ff?@  �XQ���C,����ff?��R�G��߅C%W
                                    Bx�Ov  �          @���\)?}p��j�H�p�C)Ǯ��\)?���Vff��
=C!��                                    Bx�O&  T          @����
?5�aG����\C,�����
?��R�P����Q�C%&f                                    Bx�O4�  �          @ָR���\?Y���g
=�\)C+^����\?�33�Tz����C#��                                    Bx�OCh  �          @���G�?W
=�hQ����C+k���G�?���Vff��RC#u�                                    Bx�OR  �          @���G�?Tz��h�����C+xR��G�?���Vff����C#z�                                    Bx�O`�  �          @�\)���?Q��hQ��z�C+�H���?���U��{C#�R                                    Bx�OoZ  �          @�Q�����?J=q�[����
C,+�����?Ǯ�I����ffC$�{                                    Bx�O~   �          @أ����?p���\�����C*�����?�(��H����\)C#^�                                    Bx�O��  �          @���33>���J�H��G�C/�3��33?���?\)���C(�R                                    Bx�O�L  �          @ҏ\��z�?#�
�n�R�
(�C-33��z�?�  �^�R��33C$z�                                    Bx�O��  �          @����p�?�\�i���z�C.�{��p�?�{�[���z�C%�3                                    Bx�O��  �          @�(����R>���.�R��(�C/B����R?����#33��33C)�                                    Bx�O�>  �          @��
���R>�Q��.�R��  C0@ ���R?}p��$z��Ə\C)�R                                    Bx�O��  �          @�(����\>��=p���  C/����\?�33�1G��י�C'��                                    Bx�O�  �          @�(���G�>8Q��E���p�C2���G�?h���<����(�C*E                                    Bx�O�0  �          @������>��O\)��C2�)����?fff�G���G�C*�R                                    Bx�P�  �          @����ff>���O\)���C2����ff?k��G
=���HC*�)                                    Bx�P|  �          @ҏ\���R>#�
�P  ��G�C2p����R?n{�G�����C*�q                                    Bx�P"  �          @�������>u�`  ��=qC1�{����?����U��\)C):�                                    Bx�P-�  �          @������>��S�
���C2�3����?k��K���Q�C*��                                    Bx�P<n  �          @�p�����>���j=q�z�C2}q����?��
�`����33C)��                                    Bx�PK  �          @�\)��p��#�
�~�R�G�C4p���p�?n{�xQ���RC*L�                                    Bx�PY�  �          @����z�(���{��
C:���z�>�(����R��
C/@                                     Bx�Ph`  �          @�p����þǮ��G��ffC8+�����?(������(�C-aH                                    Bx�Pw  �          @�=q������ff������C9�����?z���(��\)C-h�                                    Bx�P��  �          @�  ������������RC8������?��������RC-G�                                    Bx�P�R  �          @��H��(��������G�C9����(�?���Q����C-��                                    Bx�P��  �          @����  �8Q�����.�HC=���  >\�����1  C/#�                                    Bx�P��  �          @�=q���?��33����C.z����?�G���{����C)Q�                                    Bx�P�D  �          @�  ��@5@�33B]p�B�aH��?��R@���B�G�CxR                                    Bx�P��  �          @�녿�@E@�{B]p�B���?ٙ�@���B���C��                                    Bx�Pݐ  T          @����@9��@�z�BcB�q��?\@��RB���C	�
                                    Bx�P�6  �          @�Q쿚�H@:=q@�z�Bn�B�(����H?��H@�ffB��C��                                    Bx�P��  �          @ƸR���@C�
@��Bf=qB�p����?��@��HB�ffC ��                                    Bx�Q	�  �          @��ÿǮ@C�
@�p�B\�\B��ÿǮ?��H@�G�B���C@                                     Bx�Q(  �          @�z��ff@G
=@�p�BQ��B����ff?���@���B�z�C�)                                    Bx�Q&�  �          @���Q�@P  @���B3��B�ff�Q�@�@�
=B_C
E                                    Bx�Q5t  �          @���@  @U�@[�B��C\�@  @@�p�B>Q�C
=                                    Bx�QD  �          @��\�AG�@I��@w�B&=qC�{�AG�@33@�G�BL��CǮ                                    Bx�QR�  �          @�
=�6ff@L��@tz�B&C���6ff@�@�Q�BO33Cn                                    Bx�Qaf  �          @����2�\@!�@��RB@�HC	���2�\?�{@�
=BbC                                      Bx�Qp  �          @�  ��@@�33Be��C����?�G�@�G�B���C�                                    Bx�Q~�  �          @�ff�(��@�@�33B]\)C
� �(��?p��@���B|�C p�                                    Bx�Q�X  �          @�\)�0  @&ff@�BR  C�
�0  ?�p�@�{Bt
=Cٚ                                    Bx�Q��  �          @���1�@C33@�  BD�C^��1�?��H@�(�Bk{Cs3                                    Bx�Q��  �          @���K�@-p�@��RBA�C���K�?��@�Q�Bb=qCk�                                    Bx�Q�J  �          @ʏ\�G�@7�@�ffB?�C	p��G�?��@�G�Bb��C��                                    Bx�Q��  �          @ʏ\�A�@AG�@��B=�
C\�A�?�Q�@�G�Bc  C��                                    Bx�Q֖  �          @����0  @*�H@��RBQ{C�
�0  ?��\@��BtG�C+�                                    Bx�Q�<  �          @�p��.{@0��@��HBR��C���.{?���@�z�Bw\)C{                                    Bx�Q��  �          @��.{@G
=@�BH\)C(��.{?�Q�@��\Bp�C�                                    Bx�R�  �          @θR�B�\@W�@��\B5(�C��B�\@�@��B]�
C@                                     Bx�R.  �          @љ��R�\@_\)@�\)B,=qCO\�R�\@
�H@�  BTffC��                                    Bx�R�  $          @�z��`  @y��@��RB!C��`  @#�
@��HBK��C�                                    Bx�R.z  �          @�=q�j�H@h��@���B�CB��j�H@�@��
B?��C��                                    Bx�R=   �          @أ���(�@tz�@mp�B�C	B���(�@,(�@�33B,  C�q                                    Bx�RK�  �          @�33����@vff@aG�B{CW
����@0��@�B(��C��                                    Bx�RZl  �          @�z�����@��\@l��Bp�C������@;�@��B,��C�f                                    Bx�Ri  "          @׮��p�@��@#�
A�
=C.��p�@o\)@mp�B  C
�                                    Bx�Rw�  �          @ٙ���
=@�{@:=qA�\)C�)��
=@P  @|(�B��C
=                                    Bx�R�^  �          @ٙ����@e�@aG�A�=qC�����@�R@��B {Cp�                                    Bx�R�  �          @Å��@dz�@)��A�(�C� ��@-p�@a�B33C�                                    Bx�R��  �          @�33�~{@XQ�?�
=A��RC�)�~{@.{@2�\A�=qC�{                                    Bx�R�P  �          @�\)���\@R�\@U�B��C)���\@  @�33B%p�C�                                    Bx�R��  
�          @����G�@Tz�@���B�CG���G�@33@��B-�C�
                                    Bx�RϜ  �          @�\)���@�z�@n{A��C

���@<��@�
=B#  C^�                                    Bx�R�B  �          @�\)���H@�Q�@w
=BG�C
�{���H@2�\@��B'�C��                                    Bx�R��  �          @�\���@��@s33BffC	\)���@6ff@���B){C�                                    Bx�R��  �          @������
@dz�@b�\A��HC���
@��@�z�BffCT{                                    Bx�S
4  �          @���@e�@1�A��RCO\��@*�H@j�HB �HC�                                     Bx�S�  �          @�\)�B�\@�\)?�ffA)p�B�� �B�\@�  @�A�Q�B�8R                                    Bx�S'�  T          @�������@^{?��RAQG�CY�����@?\)@
=qA�{C��                                    Bx�S6&  �          @�(�����@]p�?��A���Cs3����@7
=@"�\A�\)C��                                    Bx�SD�  �          @�����\)@^{@n�RB	�C�f��\)@�@���B-�
C�                                    Bx�SSr  "          @�=q�~�R@z�H@Z=qA�\)Ck��~�R@333@�(�B(  C޸                                    Bx�Sb  T          @ȣ�����@N�R@_\)B=qCY�����@
=@�Q�B,z�Ck�                                    Bx�Sp�  �          @����q�@0��@u�B33C�)�q�?Ǯ@�ffB=�
C�\                                    Bx�Sd  �          @�33�c�
?�@�p�B7z�C���c�
?&ff@�G�BMC)�)                                    Bx�S�
  �          @��H�a�?޸R@�G�B=�
CǮ�a�>�G�@��BQG�C,��                                    Bx�S��  T          @���n{?�@��HB6��C!H�n{>B�\@�=qBD��C1
=                                    Bx�S�V  "          @�{�{�?���@i��B �C+��{�?333@���B4�C)�3                                    Bx�S��  "          @��\�u?��
@xQ�B)�HC0��u?
=@�
=B=p�C+G�                                    Bx�SȢ  
�          @�p��Y��?��
@�p�BPQ�CQ��Y����G�@��HB[�C5�f                                    Bx�S�H  
Z          @���e�?n{@�ffBI{C%}q�e�����@���BN  C:O\                                    Bx�S��  �          @�Q��dz�?z�H@�
=BOffC$�R�dz��(�@���BTz�C:�{                                    Bx�S��  "          @�
=�j=q?�ff@�
=BW�\C#��j=q�   @���B\�C;�{                                    Bx�T:  T          @�\)�|(�?��H@�=qB9\)C"�R�|(�����@��BB�RC5s3                                    Bx�T�  �          @�Q���p�?�(�@��B*G�C �)��p�>aG�@��B7�C0��                                    Bx�T �  
�          @Ǯ��\)?���@�(�B&G�C8R��\)?(��@���B:=qC+)                                    Bx�T/,  T          @˅��ff?��\@��HB9  C#5���ff��@�Q�BB{C5                                    Bx�T=�  T          @�z����H?k�@�G�B6C(����H��@��B:\)C9�q                                    Bx�TLx  �          @�  ���?E�@���BJ33C)Q���녿=p�@���BJ\)C>aH                                    Bx�T[  �          @�  �aG�?�{@\)B4=qC&f�aG�?��@��BJ�RC*J=                                    Bx�Ti�  �          @��H�z�H?�33@�(�B5{C#�)�z�H��@�G�B>
=C5��                                    Bx�Txj  �          @�{�p  ?W
=@W�B%{C'\)�p  �B�\@^{B*��C6�R                                    Bx�T�  �          @�����Q�?���@L(�B�C ����Q�>�p�@\��B"z�C.��                                    Bx�T��  �          @���w
=?�=q@Dz�B�HC$s3�w
==�G�@P��B \)C2T{                                    Bx�T�\  �          @���?.{�����  C���?E��B�\�0��B��                                    Bx�T�  �          @>�(�?(�����ǮBc�>�(�?��\�����X�B��                                     Bx�T��  �          @n�R>��?�(��W
=G�B��{>��@=q�1G��Bz�B��=                                    Bx�T�N  "          @\)���<���
=�mC2W
���?(���=q�[(�C��                                    Bx�T��  �          @a녿�?�Q�� ���D�C�{��?�����\)C��                                    Bx�T�  �          @l(����R?����1G��O��Cz῾�R@��
=q�  B�33                                    Bx�T�@  �          @%�>�
=>��H�
=k�BD(�>�
=?�
=�ff�m  B�u�                                    Bx�U
�  �          @E�>�?p���6ff��B33>�?�G����T�HB��                                    Bx�U�  �          @&ff?��?Y���
=�)Bd�?��?\�   �K�HB���                                    Bx�U(2  �          @����Q�?�33� ���lBօ��Q�?ٙ������((�B˞�                                    Bx�U6�  
(          ?�\��  ?fff��z��a{B��
��  ?��������B��f                                    Bx�UE~  �          @$z���H?�ff���F��B�Ǯ���H@�
�����  B΅                                    Bx�UT$  T          @J�H=�G�?�\)�+��k=qB�ff=�G�@
=��
�$(�B�Ǯ                                    Bx�Ub�  �          @QG����?���+��^\)Bͳ3���@!��   �=qB�\)                                    Bx�Uqp  �          @n{��G�?���[��RB����G�@Q��5�H
=B�W
                                    Bx�U�  �          @`�׾��H?��H�H���z�B�z���H@ff�"�\�:33B˽q                                    Bx�U��  �          @W
=��?�=q�8Q��p{B�.��@������*�RB�
=                                    Bx�U�b  �          @g���Q�@��8Q��V\)B�  ��Q�@:=q���p�B���                                    Bx�U�  �          @g
=��(�?�{�(����B����(�?�\)�
�H�B
=Bͨ�                                    Bx�U��  T          @XQ��ff�333�2�\�jffCL+���ff>L���7��up�C,�\                                    Bx�U�T  "          @?\)��\)?����\)�g{C� ��\)?����G��-��B�8R                                    Bx�U��  �          @#�
�Y��?��Ϳ�Q��Jz�B�=q�Y��?�׿��
(�B�\)                                    Bx�U�  �          @b�\�\@�R�%�Cz�BǙ��\@:�H��G���
=B�                                      Bx�U�F  T          @\�;#�
@Q�����733B��=�#�
@A녿�=q����B��                                    Bx�V�  T          @z�H���
@/\)�2�\�5�HB�Q콣�
@^{�����B���                                    Bx�V�  �          @~�R���
@(��Dz��M�B�녾��
@Q��	���ffB��                                    Bx�V!8  
�          ?�=q�L��?�(���ff�9�B�k��L��?Ǯ�Y����(�B�ff                                    Bx�V/�  �          @Vff����@�\�%��Lz�B�\����@0  ��ff���B�u�                                    Bx�V>�  "          @(�ÿ�?���(��b�B�W
��?��H��33�z�Bѣ�                                    Bx�VM*  T          @9���   ?����\)�r
=B�\)�   @녿�Q��+\)B�k�                                    Bx�V[�  T          @Mp���ff?����&ff�X��B����ff@#33�����\B�
=                                    Bx�Vjv  �          @n{�k�@���4z��F�B��k�@J�H��z����\B�(�                                    Bx�Vy  �          @_\)��ff@=q�{�5(�B�W
��ff@Dz��=q���
B���                                    Bx�V��  
�          @S33�G�@Q���� �
B�B��G�@<(���  ���HB�Ǯ                                    Bx�V�h  "          @Q녿(�@{���5{B��)�(�@6ff���R��G�B�L�                                    Bx�V�  �          @g
=�W
=@=q�(���=��B��f�W
=@HQ��p���
=B���                                    Bx�V��  T          @b�\<#�
@���"�\�8{B�aH<#�
@H�ÿ�\)��z�B��                                    Bx�V�Z  �          @g�>�\)@#33�"�\�2��B�\)>�\)@N�R��=q��{B�                                      Bx�V�   T          @]p�>���@ ���z��*\)B���>���@G��������B��f                                    Bx�Vߦ  �          @A�>W
=@	�����1��B�{>W
=@.{�����  B�\)                                    Bx�V�L  �          @i��=��
@��/\)�DG�B�\=��
@HQ��=q���
B�                                    Bx�V��  "          @��<#�
@333�J�H�B\)B���<#�
@j�H����B��3                                    Bx�W�  	�          @��<#�
@<���G
=�:33B���<#�
@s33���R��
=B��R                                    Bx�W>  �          @���=�G�@5��7
=�5\)B�aH=�G�@fff�����G�B�aH                                    Bx�W(�  �          @�Q�>u@AG��>�R�2\)B��R>u@tz����33B��                                    Bx�W7�  �          @�(��aG�@Dz��+��$p�B�W
�aG�@qG������
=B�Ǯ                                    Bx�WF0  �          @z=q�&ff@E�����B��f�&ff@h�ÿ������
B�B�                                    Bx�WT�  �          @$z᾽p�@녿�\)���
B�33��p�@!G����
��33B�z�                                    Bx�Wc|  �          @z�H���
@L(�����B��
���
@q녿�z���ffB�ff                                    Bx�Wr"  �          @���>aG�@qG��j�H�0��B�u�>aG�@�����R��(�B���                                    Bx�W��  "          @�=q>��
@���[�� �\B��>��
@��R��z����B��R                                    Bx�W�n  T          @�33>�ff@5��p���R�B�Ǯ>�ff@z=q�'��p�B�                                    Bx�W�  �          @�33>�p�@\)�N{�P{B��>�p�@Z=q�{���B���                                    Bx�W��  �          @�{>Ǯ@)���Mp��H��B�B�>Ǯ@c�
�	����{B��                                     Bx�W�`  �          @���>�p�@.�R�:�H�:��B��=>�p�@b�\������B�\                                    Bx�W�  �          @��>.{@;��:�H�3=qB�u�>.{@o\)���
���
B��)                                    Bx�Wج  �          @��>8Q�@z��W
=�]�B���>8Q�@S33�����RB���                                    Bx�W�R  �          @S�
=�G�@(��(��@�B�  =�G�@7�������
=B�k�                                    Bx�W��  
�          @`  ��G�@��&ff�B�
B��{��G�@AG���Q���(�B�8R                                    Bx�X�  �          @I���#�
@�
�
=�CQ�B�
=�#�
@.�R�����
=B��
                                    Bx�XD  �          @e���@���&ff�;33B����@G���z���
=B�Q�                                    Bx�X!�  T          @w
=�#�
@���:�H�D33B�(��#�
@Q녿�
=��B���                                    Bx�X0�  �          @n�R�&ff@�\�7
=�I�Bӊ=�&ff@G���
=���
B�p�                                    Bx�X?6  �          @x�ý�G�@ ���=p��G{B��ͽ�G�@W
=������\)B��{                                    Bx�XM�  �          @:=q=�G�?��G��9��B�B�=�G�@\)��G���(�B���                                    Bx�X\�  �          ?�
=?   ?�녿��H�ffB��\?   ?�(��+�����B���                                    Bx�Xk(  �          @{?J=q?��Ϳ�z��{B��?J=q@{�333����B��                                    Bx�Xy�  �          @��?�R?Ǯ���!�B��\?�R?����O\)��G�B���                                    Bx�X�t  �          @;�?(��@��   �,z�B��
?(��@%���Q�����B�B�                                    Bx�X�  �          @�z�?J=q@'
=�E�B=qB�33?J=q@`  �G����B�z�                                    Bx�X��  �          @c33?333@\)�(���AffB��=?333@@  ��(���\)B��                                    Bx�X�f  �          @~{?s33@ ���8Q��;�B���?s33@U�������\B�Q�                                    Bx�X�  �          @��?Tz�@=q�dz��Z�B�Ǯ?Tz�@^�R�"�\���B�
=                                    Bx�XѲ  �          @���?Q�@Q����R�r
=B���?Q�@p���XQ��%�B�\)                                    Bx�X�X  �          @�{?��@$z���G��y�B��R?��@�(��w��+�B��                                     Bx�X��  �          @�>�{@!���=q�}  B��f>�{@���y���-p�B��                                    Bx�X��  �          @���>�p�@���Q�ǮB�#�>�p�@\)�����8
=B��                                    Bx�YJ  �          @��R�0��@����B�33�0��@xQ����Az�B�G�                                    Bx�Y�  �          @�=q��{?�(���(�aHB��
��{@l����(��BQ�B�u�                                    Bx�Y)�  �          @�z��,����>�
=@�ffCp\�,����G����H�O33CoL�                                    Bx�Y8<  �          @�=q�u�����  �:��CMB��u������33�N��C633                                    Bx�YF�  �          @�����333�J=q��=qCP��������{���CD��                                    Bx�YU�  �          @Ӆ��z���\�y���\)CJ����z������&(�C:�H                                    Bx�Yd.  �          @�����  ��=q��  �;\)CEc���  >Ǯ��p��D�C.�R                                    Bx�Yr�  �          @���~{�!G������R��C=  �~{?��R��p��L�HC"��                                    Bx�Y�z  �          @���i���z������`�HC=\�i��?�\)��z��XffCn                                    Bx�Y�   �          @�=q�l(���������^G�C:.�l(�?�  ���R�Rz�C�\                                    Bx�Y��  �          @���}p�>Ǯ��ff�RG�C.^��}p�@
�H����9z�C@                                     Bx�Y�l  �          @����Q�=�Q��E��\)C3���Q�?�z��7���  C'�\                                    Bx�Y�  �          @�G��A�?
=����lp�C)��A�@ff��(��I�C33                                    Bx�Yʸ  �          @��R�,��?J=q��ff�qz�C#���,��@{����H  C	��                                    Bx�Y�^  �          @��R�   ?�=q���R�v{C�   @C�
��p��@C.                                    Bx�Y�  �          @�
=��@/\)���jffC�H��@�z���33�%B��                                    Bx�Y��  �          @��
�k�?����Q�(�B�
=�k�@�����{�RffBͨ�                                    Bx�ZP  �          @�z��\)@(����=q�w�RB�k���\)@����Q��/  B��                                    Bx�Z�  �          @ۅ����@!���33�{��B�aH����@�Q���=q�2�HB��                                    Bx�Z"�  �          @�  ���\?�33��33k�B�ff���\@�Q������M�B�{                                    Bx�Z1B  �          @��ÿ�p�@   ��G��t�
B�ff��p�@�G������+��B���                                    Bx�Z?�  �          @�zῐ��@
=q��33  B�{����@z=q����8�HB�.                                    Bx�ZN�  �          @�
=�+�?��H���\B�p��+�@z=q��G��I�B�Q�                                    Bx�Z]4  �          @��
�Q�?��
��(�  B�p��Q�@`����ff�W�
B�aH                                    Bx�Zk�  �          @����^�R?�G�����z�B�녿^�R@\������WG�B�=q                                    Bx�Zz�  �          @ᙚ�aG�@ff��{ffB�
=�aG�@�\)��{�B��B�=q                                    Bx�Z�&  �          @�
=��ff?�33���
Q�B�녿�ff@�\)��ff�Xz�B�z�                                    Bx�Z��  �          @��
��{@���ff��B�33��{@�(��ƸR�P�RBԮ                                    Bx�Z�r  T          A��\@
=���\�=B��׿\@����\)�L�B��                                    Bx�Z�  �          AG�����@�H���Hu�B��
����@�����
=�K��B�                                      Bx�Zþ  �          @��R���\@"�\��ff��B�녿��\@�������G  B�                                    Bx�Z�d  �          @��H�@  @%���=q\B�LͿ@  @��H��ff�@G�BŅ                                    Bx�Z�
  �          @�\)��  @ ���ָR(�B�aH��  @�����
�?�HB�Ǯ                                    Bx�Z�  �          @�ff��=q?����H  B���=q@�����
=�O�B��                                    Bx�Z�V  �          A��?�z�@�
=� ���p��B���?�z�@����p��z�B�\)                                    Bx�[�  �          A%��?���@�=q���t��B�G�?���@�  �׮�"�B���                                    Bx�[�  �          A�H?u@���(�W
B�W
?u@����ff�1�B�ff                                    Bx�[*H  �          A1?��@�  ����y�HB�B�?��A\)��z��'  B�#�                                    Bx�[8�  �          AE�?�\@\�'��kQ�B��{?�\A
=��\�=qB�Ǯ                                    Bx�[G�  �          A<(�?�z�@��H���^�B��=?�z�A�\��G����B��                                    Bx�[V:  �          A@z�?��@�\)�$���kB�\)?��A����ff�=qB�                                      Bx�[d�  �          AB�R?��@Ǯ�$���f��B���?��Az���33���B��R                                    Bx�[s�  �          AB�R?��
@����$Q��f33B�L�?��
A������B��)                                    Bx�[�,  �          AB=q?�ff@˅�#
=�d(�B�p�?�ffA��ff��\B���                                    Bx�[��  �          AB�\?�@�G��#��e�B��)?�A���Q���HB��=                                    Bx�[�x  �          AE?��@�ff�%��d\)B��=?��AQ����H��B�=q                                    Bx�[�  �          AG
=?�ff@�=q�&�R�cQ�B�ff?�ffA=q��\�Q�B�k�                                    Bx�[��  �          AH��?�@ҏ\�(Q��d�B�B�?�A
=����
=B�ff                                    Bx�[�j  �          AF�R?���@Ϯ�'
=�d��B���?���AG����
��B�                                      Bx�[�  �          AE��?޸R@�33�'33�g33B�\)?޸RA\)����B��3                                    Bx�[�  T          AF{?�33@�33�'�
�g�HB��H?�33A�
��R�Q�B�                                    Bx�[�\  �          AG
=?޸R@��
�(���g�RB��?޸RAz���  �=qB��)                                    Bx�\  �          AG�?˅@�=q�*=q�j�B���?˅A(����G�B��                                    Bx�\�  �          AHz�?�(�@˅�+
=�j\)B���?�(�A���z��33B���                                    Bx�\#N  �          AH��?�
=@�z��(  �c�\B��\?�
=A (���33���B��                                    Bx�\1�  �          AK�?�@���*{�b\)B�\)?�A#�����=qB�aH                                    Bx�\@�  T          AO
=?�@����-��c  B�?�A&{���B��3                                    Bx�\O@  �          AQp�?���@�Q��0���fz�B�p�?���A%p������\B��q                                    Bx�\]�  �          AQp�?�=q@׮�1��g  B���?�=qA%G����\���B��                                    Bx�\l�  �          AR=q?���@׮�2{�gz�B�=q?���A%����(��ffB��                                    Bx�\{2  �          AR{@ff@����3
=�i�B�k�@ffA"�H� (��p�B��q                                    Bx�\��  �          AIG�@p�@�p��+��jQ�B��{@p�A�H���R�Q�B�G�                                    Bx�\�~  T          AG�@Q�@�z��)�h�RB��3@Q�A���(�B�=q                                    Bx�\�$  T          ADQ�@�@��
�&=q�g��B��3@�A(������
B�#�                                    Bx�\��  �          AF�H@�@���(���h��B�
=@�A��G���RB�u�                                    Bx�\�p  �          AMG�?�(�@�{�0���nQ�B�?�(�AG��   �\)B�aH                                    Bx�\�  �          AO
=?���@�
=�6�H�xz�B�G�?���A���Q��$�B��
                                    Bx�\�  �          APz�@�@�\)�8(��x\)B�\@�AG��	G��$�
B�u�                                    Bx�\�b  �          AP��?��@����:�R�}\)B�k�?��A\)����)Q�B�p�                                    Bx�\�  �          AP��?�=q@�\)�<���B�aH?�=qA��z��/Q�B��                                    Bx�]�  �          AM��@��@�
=�<Q�\)BwQ�@��A(��z��9�B��\                                    Bx�]T  �          AMp�@%@w
=�?\)� B`\)@%A Q���H�C�HB�#�                                    Bx�]*�  �          AN�\@'�@�\)�>�R�3Bh�H@'�Ap��  �=z�B�#�                                    Bx�]9�  �          AQG�@#�
@��H�=�z�BxG�@#�
A=q���3��B��
                                    Bx�]HF  �          AV=q@#33@��
�AffB~(�@#33A  �p��0�B�.                                    Bx�]V�  �          AV{@�R@���G\)z�Bl�
@�RA�
� Q��B=qB�p�                                    Bx�]e�  �          AX(�@�R@����J�\33Bv�R@�RA���#\)�Dp�B��R                                    Bx�]t8  �          AY�@�R@n{�L��z�BaG�@�RA��(  �J�B�ff                                    Bx�]��  �          A[�@0  @g
=�O33aHBS
=@0  A�R�*�R�LG�B��)                                    Bx�]��  �          A]��@=p�@Mp��R=q8RB=�@=p�@��
�0  �R\)B�                                    Bx�]�*  �          A^�\@@  @7
=�TQ�Q�B.G�@@  @�33�4  �X{B��H                                    Bx�]��  �          Ab�\@[�@>{�V�R��B#ff@[�@�Q��5p��T�
B�L�                                    Bx�]�v  �          A`��@N�R@B�\�U�(�B-Q�@N�R@����3��T=qB�
=                                    Bx�]�  �          A_�@Dz�@'
=�V{� B!p�@Dz�@�p��7
=�[�RB�
=                                    Bx�]��  �          Aj�H@Q�@����Z�Hk�BK�@Q�A���2�H�GQ�B�\)                                    Bx�]�h  �          Ai�@Q�@����W
=ǮBX=q@Q�A��,Q��>�
B���                                    Bx�]�  �          AhQ�@QG�@���U��B[�
@QG�A��*{�<ffB�Q�                                    Bx�^�  �          Ahz�@R�\@���X���\BMG�@R�\A{�0z��EQ�B�\)                                    Bx�^Z  �          AiG�@[�@{��Y{BC�R@[�A  �2ff�G�B�8R                                    Bx�^$   �          Ah��@aG�@�{�W�ǮBG�H@aG�A
=�.�R�B��B�                                      Bx�^2�  �          Ahz�@xQ�@���U��fB@  @xQ�A  �+��>\)B�L�                                    Bx�^AL  �          Ag�@j=q@�Q��QG�=qBWz�@j=qA���$  �4�B��                                    Bx�^O�  �          Ag�@g�@�\)�Q����BW�@g�Az��$Q��4��B�k�                                    Bx�^^�  T          Ac33@P��@�G��D���p\)BzQ�@P��A'33����=qB�W
                                    Bx�^m>  �          Ab�R@AG�@�=q�B�H�m33B��\@AG�A*�R����B�W
                                    Bx�^{�  �          Ab�H@C33@�=q�B�H�m  B�G�@C33A*�R����B�#�                                    Bx�^��  �          Ab�R@;�@��
�B�\�l��B�(�@;�A+\)�z��G�B�W
                                    Bx�^�0  �          AaG�@P  @�\)�F�H�xG�Bq�H@P  A   ��&�RB�                                      Bx�^��  �          Aap�@R�\@�G��F=q�v��Bq�@R�\A ������%ffB��R                                    Bx�^�|  �          A`Q�@[�@��
�J�H��B[=q@[�A����\�4
=B�p�                                    Bx�^�"  �          Aap�@fff@����J�\  BYz�@fffA�H�G��1
=B�8R                                    Bx�^��  �          Ab�H@n{@�=q�L���BQQ�@n{A��� z��433B�\)                                    Bx�^�n  �          Ac�
@n�R@��H�LQ��~��BV�@n�RAz��ff�0(�B�=q                                    Bx�^�  T          AeG�@q�@�G��M����BTz�@q�Az�� (��133B��3                                    Bx�^��  �          Ae�@n{@��
�Mp��~��BX  @n{A���33�0{B��{                                    Bx�_`  �          Adz�@\(�@����K33�{p�BhG�@\(�A�R��\�*�B��3                                    Bx�_  �          Aa��@^{@��J=q�~��B`@^{AG���
�.�
B�.                                    Bx�_+�  �          Ac\)@U�@���J{�{Q�BlQ�@U�A�H�G��)�
B��f                                    Bx�_:R  �          AdQ�@_\)@���MG�33B_�R@_\)A=q��R�0=qB�.                                    Bx�_H�  �          Ad��@u@�
=�M����
BQff@uA�� (��1��B��f                                    Bx�_W�  �          Adz�@tz�@�Q��O�
BF��@tz�A���$���8��B�k�                                    Bx�_fD  �          Ac�
@x��@���Pz�B;�
@x��A���'��=��B�L�                                    Bx�_t�  �          AZ�R@s�
@vff�I��B5  @s�
A���#\)�@=qB��H                                    Bx�_��  h          AT  @�HA#����\�B�Q�@�HAF�H�W
=�mB��                                    Bx�_�6  �          AM@
=A$Q�����
=B�@
=AE��=p��T��B��=                                    Bx�_��  �          AL��@Tz�A�\�G��"�B��@Tz�A:=q��\)��  B��                                    Bx�_��  T          AJff@XQ�Az��\)�'�
B��
@XQ�A5p������=qB�                                    Bx�_�(  T          AK33@i��AQ��
�\�,=qB��{@i��A2�R��������B���                                    Bx�_��  
�          ALQ�@5�A
=�
�\�*�B��
@5�A9������\)B�z�                                    Bx�_�t  �          AN�R@2�\A�\�
�R�(��B��@2�\A<Q�������=qB�W
                                    Bx�_�  �          AP��@<(�A�R�G��*  B�z�@<(�A=p���p����B�#�                                    Bx�_��  �          AU�@9��A����R�-33B�B�@9��AAG���{���B���                                    Bx�`f  �          AS�@O\)A�R��H�<{B��
@O\)A7�������
B�ff                                    Bx�`  �          AT��@P  A�
�33�;z�B��@P  A8������̏\B��{                                    Bx�`$�  �          AX��@Tz�AG��"�\�BQ�B��=@Tz�A9���
=�ڣ�B�                                      Bx�`3X  �          A]p�@P  A	p��%���A�RB��\@P  A>{�ə��؏\B�k�                                    Bx�`A�  �          A0  @!�@|���\)�yffBeQ�@!�@�=q�أ��(=qB��\                                    Bx�`P�  �          A
=?�p���R���\�)C��\?�p�@�H��
=u�Bj�                                    Bx�`_J  T          AQ�@	��?���(�p�Bff@	��@�p������[  B��=                                    Bx�`m�  �          A�?�Q�    ���� =��
?�Q�@c�
�	��Bu�R                                    Bx�`|�  |          AB�H>B�\�����3
=  C�33>B�\=����@��±��A�G�                                    Bx�`�<  "          AQ�?ٙ��.{�H  .C�?ٙ�@���Ip��BS�
                                    Bx�`��  �          AO�?��
�Q��I�z�C�j=?��
@'��I��3B_�                                    Bx�`��  �          AO�@P  ���H�G��C�Ф@P  @i���?�
��BA(�                                    Bx�`�.  �          AMp�@:=q�xQ��G\)L�C��f@:=q@w
=�>{�3BS��                                    Bx�`��  �          Ac�
@{�?���Y�k�A
�H@{�@�Q��E���r33B^�                                    Bx�`�z  �          Ac�@W
=����\��p�C���@W
=@����Nff�{B^33                                    Bx�`�   �          Aa�@QG��8Q��[33�C�@QG�@�=q�O
=BY��                                    Bx�`��  �          Ab{@E�s33�\  �C�u�@E@�(��Q�aHB[(�                                    Bx�a l  �          Adz�@5��
=�^{=qC��\@5@p���W�u�BS��                                    Bx�a  �          Ac�@,���ff�\  ��C�w
@,��@E�Yp�BCff                                    Bx�a�  �          Ae�@1���\�]�HC�H�@1�@L(��Z�H��BC�
                                    Bx�a,^  �          As�
@L��?u�j�\\)A�@L��@�  �R�H�t��B=q                                    Bx�a;  T          Az�H@S�
@?\)�pQ��B(G�@S�
A(��K��Y\)B��                                     Bx�aI�  T          Ayp�@Y��@��p��  B�@Y��@����Q��d
=B���                                    Bx�aXP  �          A{�@P��@�
=�l(�G�BQp�@P��A���@Q��G�HB�\)                                    Bx�af�  �          A{�@P��@|���m  BI��@P��Ap��C��L
=B�z�                                    Bx�au�  �          A|��@;�@�ff�b{�}p�B�8R@;�A5��)G��(
=B�#�                                    Bx�a�B  �          A|  @*=q@�z��`z��{��B�ff@*=qA8(��&�\�%��B�                                      Bx�a��  �          Ay@=p�@�{�^=q�~B��)@=p�A0���'33�)�RB���                                    Bx�a��  �          Av�\@g
=@��H�Pz��gp�B33@g
=A<z����33B��H                                    Bx�a�4  �          Av�H@p  @���P  �eB|��@p  A=���R��HB���                                    Bx�a��  �          Ax(�@i��@���R�R�h�B}�
@i��A<�����\B���                                    Bx�à  �          Aw
=@~�R@���R�\�i��Bq�@~�RA8z��\)�=qB���                                    Bx�a�&  �          Au��@���@�{�V�R�s�
B`�
@���A-�� Q��#  B�=q                                    Bx�a��  T          Az=q@�
=@���`  ��BP\)@�
=A%��-���0p�B��q                                    Bx�a�r  �          A�@�p�@����k�u�B#�@�p�A=q�@���C�
B~Q�                                    Bx�b  �          A~�H@�33@�33�j�Hu�B&��@�33A�R�?�
�CQ�B�                                    Bx�b�  �          A~�H@�Q�@�Q��j�R(�B-Q�@�Q�A���>�R�A�RB�p�                                    Bx�b%d  �          A
=@�33@����iG�(�B%�@�33A���=��?ffB|=q                                    Bx�b4
  �          A�@�  @���g��HB)z�@�  A(��9�:�\B{�                                    Bx�bB�  �          A}��@�ff@�\)�g
=u�BGff@�ffA"=q�6�H�8��B�                                      Bx�bQV  �          A~{@��
@�ff�g
=�HBBQ�@��
A!�7
=�8\)B�G�                                    Bx�b_�  �          A}p�@�  @�G��f�\�B;(�@�  A33�7��9�\B�Q�                                    Bx�bn�  �          A{33@u@����h��L�BC��@uAff�<(��B33B���                                    Bx�b}H  �          Ax  @qG�@���f=qW
BDp�@qG�A���9��B(�B��)                                    Bx�b��  �          Ayp�@�33@��H�e��B@��@�33A��7��=z�B�W
                                    Bx�b��  �          Az{@�\)@�z��d����B>z�@�\)Az��7
=�<(�B�(�                                    Bx�b�:  T          A|(�@�p�@����g�Q�B@z�@�p�A���9G��=(�B��                                    Bx�b��  �          Au�@�@�(��`��#�B?��@�A�\�333�;{B�8R                                    Bx�bƆ  �          As�
@|��@�(��W��y(�B`�\@|��A)���#33�(�B�.                                    Bx�b�,  �          Ak33@�
=@����V=qL�B1��@�
=A\)�,Q��=�\B��{                                    Bx�b��  �          Ad��@�Q�@��O
=�A�(�@�Q�@�R�0Q��J(�BR\)                                    Bx�b�x  �          A`z�@�z�@�33�Ep��z�B,�@�z�A  ���1�BxG�                                    Bx�c  �          AZ�R@l��@�{�6ff�c�Bp�\@l��A%�� ���G�B��{                                    Bx�c�  �          AX��@n{@�33�3�
�`  Br�\@n{A&�\��33�z�B��R                                    Bx�cj  �          AY@.{@��H�6�R�d�HB���@.{A+
=����
B��                                     Bx�c-  �          A]@L(�@����/\)�R
=B�\)@L(�A6=q��z���
=B��                                    Bx�c;�  �          A[\)@-p�@�p��H(�\)BoQ�@-p�A�Q��733B��
                                    Bx�cJ\  �          A]�@(�@����M��.Bs��@(�A�
�"�H�>�B���                                    Bx�cY  |          A_�@�
@�z��E��~��B�� @�
A�H���(�B��
                                    Bx�cg�  T          A_\)@���?����S�#�Af=q@���@\�<���g33B[�                                    Bx�cvN  �          A_
=@��?�\)�R�H  A�G�@��@�G��8���_�BdG�                                    Bx�c��  �          A[�@|��?�z��P����A��\@|��@����6�\�`�HBk\)                                    Bx�c��  h          AY@Z�H?Ǯ�Pz��
A���@Z�H@�{�7
=�f  Bx33                                    Bx�c�@  h          Ac�
@��@�{�H���~33BC{@��A=q����1G�B��=                                    Bx�c��  �          Al��@�G�@�=q�L  �p�HBaff@�G�A'�
�33� 33B��)                                    Bx�c��  �          Ar=q@hQ�Ap��<  �K�B�\)@hQ�AJ�\��
=��z�B�                                    Bx�c�2  �          Ar�\@K�A33�3
=�>z�B���@K�AS�
���
��p�B���                                    Bx�c��  �          Aq�@0��A*�R�$(��+B�
=@0��A\����z����
B�aH                                    Bx�c�~  �          Ap�׿��A<(���H�\)B�G����Adz��p  �j�RB�z�                                    Bx�c�$  �          At�Ϳ��A
=�;�
�J��Bʏ\���AS\)����B�8R                                    Bx�d�  �          Ar{@��A���8z��FQ�B�p�@��AS�
��  ��(�B���                                    Bx�dp  �          As�
@(�Ap��@  �P\)B���@(�AO�
����Q�B��f                                    Bx�d&  T          As�@��A�1���=p�B���@��AN=q����(�B���                                    Bx�d4�  �          As
=@z�HA�2ff�D  B�Q�@z�HAG33�����33B�{                                    Bx�dCb  �          A[�@�G�?��J�\u�@�ff@�G�@��\�8  �cz�B4�
                                    Bx�dR  |          Aw�@����p��.=q�I=qC�k�@���,���NffaHC�8R                                    Bx�d`�  �          A\)@����G\)���33C�<)@����=q�N�\�X  C��)                                    Bx�doT  �          A
=@����K�����  C�G�@����  �LQ��U=qC�Y�                                    Bx�d}�  
)          A�33@�ff�J=q�\)�z�C�1�@�ff��N{�U\)C���                                    Bx�d��  �          A��@�Q��P�������C�f@�Q��\)�R=q�V\)C�f                                    Bx�d�F  �          A��@�  �G��F�H�D  C�Ǯ@�  ����rff�RC�G�                                    Bx�d��  T          A���A(������Y���\ffC�O\A(�?aG��ep��o��@Å                                    Bx�d��  �          A�(�@�����H�[��lz�C��@��?}p��g33\A��                                    Bx�d�8  �          A
=@ə��l���`(��yC���@ə�?��e�3A�G�                                    Bx�d��  �          A�
@أ��  �dQ��}��C�˅@أ�@W
=�`���v��A�
=                                    Bx�d�  �          A�
@�33�G��c\)�}=qC��=@�33@c33�^ff�s�A���                                    Bx�d�*  �          A{�@ڏ\?&ff�b=q8R@���@ڏ\@�\)�MG��Z�
B$��                                    Bx�e�  �          A{33@��H>��
�g33=q@@  @��H@����S�
�f\)B.\)                                    Bx�ev  
�          A{�
@\>���h  \)@�ff@\@��R�S��d��B1��                                    Bx�e  �          A|��@��?�  �f�H�=A[
=@��@�(��LQ��X�B@�                                    Bx�e-�  �          A�ff@�\)@=q�l(�L�A���@�\)@��H�K��R��BWff                                    Bx�e<h  �          A�
=@���@xQ��mG���Bp�@���A��D  �Ep�Bu�                                    Bx�eK  �          A�Q�@���@�G��o\)�=B'ff@���Aff�C\)�B\)B~�                                    Bx�eY�  �          A�\)@��H@����o�
�
B�R@��HA
=�EG��B�Br{                                    Bx�ehZ  T          A��
@���@��H�n�\�RB$
=@���A�\�@���<{Bw��                                    Bx�ew   �          A�(�@�ff@|���n=qBff@�ffA���Dz��@{Be�                                    Bx�e��  �          A��@���@�=q�m��aHB33@���A�@  �:��Br��                                    Bx�e�L  |          A��@��
@��R�iG����B0\)@��
A!��9�6�B|��                                    Bx�e��  �          A�@��H@p��o�
�RA�@��H@�\)�Pz��R
=BJz�                                    Bx�e��  �          A�
=@�=q�Y���f�\�~�
C��3@�=q@�p��Zff�h�RB�                                    Bx�e�>  �          A~{@��
@�  �c��(�B�
@��
A���7\)�933Br�                                    Bx�e��  �          A|��@��H@S33�fffp�A�\@��HA(��A��Gz�Bb                                    Bx�e݊  �          A}�@�p�@8���g��=A��@�p�Aff�D���LffB\�R                                    Bx�e�0  �          A|��@��@W
=�e�A�ff@��Az��?�
�E{B^�                                    Bx�e��  �          A�p�@У�@���jff�A���@У�@�(��K\)�N�RBE�                                    Bx�f	|  |          A�@�  ?����o
==qAQp�@�  @�33�S�
�V
=B9��                                    Bx�f"  �          A�=q@ָR?�{�pz��3AXz�@ָR@�p��T���VQ�B;�                                    Bx�f&�  �          A�z�@ڏ\?�
=�pz�z�Az�@ڏ\@أ��W��Y��B2�H                                    Bx�f5n  �          A�{@���?\)�k�
z�@�ff@���@\�W
=�_�\B'��                                    Bx�fD  �          A�(�@��ÿ���i
=C�h�@���@��R�_\)�lB�                                    Bx�fR�  T          A�\)@���Q��ep��y�C��{@��@[��ap��r\)A���                                    Bx�fa`  T          A��@�Q��p��a�s��C��\@�Q�@QG��^ff�n  A�z�                                    Bx�fp  T          A�{@ٙ�>u�g���@�\@ٙ�@�p��U��az�B\)                                    Bx�f~�  T          A�33@�z�?�(��o33��A_�
@�z�@߮�T���^
=BG�                                    Bx�f�R  �          A�ff@��H>���p(�#�@�@��H@�(��\���g�RB+ff                                    Bx�f��  
�          A�
=@�=q>L���t��\)@G�@�=q@�{�a�m�HB6p�                                    Bx�f��  �          A��@�
=�&ff�t��#�C��H@�
=@����f�H�uffB#                                      Bx�f�D  T          A�ff@�  �N�R�k
=B�C�޸@�  @!��mG���A��H                                    Bx�f��  �          A�=q@�z���2�H�/
=C�P�@�z����R�_��o�C�Q�                                    Bx�f֐  �          A�G�@�
=��\�L  �P  C�>�@�
=�\)�j�\�C�s3                                    Bx�f�6  �          A�Q�@�\)���\�]���n��C��{@�\)?�z��e��~��A6�\                                    Bx�f��  �          A|  @�z��y���_
=�y��C���@�z�?�{�f=qffAj�H                                    Bx�g�  T          Ar�R@�\)�o\)�T���~
=C��R@�\)?\�[�

=Aw\)                                    Bx�g(  T          At��@�\)�Tz��R�\��C�Ǯ@�\)?����W3333A���                                    Bx�g�  �          Av=q@����(��d  Q�C��{@��@��V{�uQ�B)Q�                                    Bx�g.t  |          Ax��@��\��ff�Zff�y��C�P�@��\?=p��f�\@�{                                    Bx�g=  T          A��@���?�(��h  33A#
=@���@љ��P  �X=qB0
=                                    