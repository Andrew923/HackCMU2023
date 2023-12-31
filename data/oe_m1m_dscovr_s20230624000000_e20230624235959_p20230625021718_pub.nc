CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230624000000_e20230624235959_p20230625021718_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-25T02:17:18.860Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-24T00:00:00.000Z   time_coverage_end         2023-06-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��5   �          A�R@�z���z�@���Bp�C��)@�z�����@�G�A��\C�~�                                    Bx��C�  �          A\)@�����H@�\)B�C��f@����G�@��
A�33C��                                    Bx��RL  �          A33@��
��
=@��
B�C���@��
��\)@��A�
=C�j=                                    Bx��`�  
�          A
=@�������@�{B{C��R@�����
=@�z�A���C���                                    Bx��o�  
�          A�
@�ff�ڏ\@�=qA���C�B�@�ff��Q�@�Q�A��HC�f                                    Bx��~>  �          A�
@��H��  @��B�HC�T{@��H��
=@�z�A�z�C��                                    Bx����  
�          A�\@W����H@�z�BC���@W���p�@��B Q�C�8R                                    Bx����  �          A=q@0������@�z�B&\)C�G�@0����Q�@�33B��C�
                                    Bx���0  "          A(�?�{��\)@ϮB,�C��)?�{��@�B�RC��)                                    Bx����  "          A�?�����  @��B
=C��R?�����\@w�A��
C�y�                                    Bx���|  T          AG�@L(���@��A�  C��R@L(���{@W�A�(�C��{                                    Bx���"  "          A@S33��33@S33A��C�e@S33�  @�AW33C��f                                    Bx����  T          A@	����Ϳ@  ���
C���@	�������
�K�
C��3                                    Bx���n  "          A\)@Q�����G���C���@Q��(��3�
����C���                                    Bx��  �          A��?���{�(�����RC��R?���
ff�{���
=C�Ф                                    Bx���  �          A�?����{�\)�q�C�K�?����
�H�r�\��z�C���                                    Bx��`  T          A  ?O\)���Vff���
C��)?O\)�  ���H��C��=                                    Bx��.  �          A�>W
=�
�R��������C���>W
=�   ��
=�z�C���                                    Bx��<�  �          A�þ�p��33��{�݅C��H��p���\)�����C���                                    Bx��KR  	�          A�?���33��{�3�
C��H?���G��J=q��ffC��R                                    Bx��Y�  T          A33@�=q�
=?��
A/�C�5�@�=q�?
=@g�C���                                    Bx��h�  �          A=q@�����  @��RA��C�33@������@J�HA��
C�1�                                    Bx��wD  "          A�@�Q����@6ffA�z�C��@�Q���?�
=Ap�C���                                    Bx����  
�          AG�@��\��?�
=AC�/\@��\�
{>�@*�HC���                                    Bx����  T          A�H@��R�33?
=@\(�C���@��R�33�.{�z�HC���                                    Bx���6  �          A Q�@�=q���?�{A+�C��\@�=q��
?!G�@hQ�C�o\                                    Bx����  �          A (�@�  �
=q?�Q�A�C�j=@�  ���>�@'�C�4{                                    Bx����  �          A   @�Q����?�Q�A��C���@�Q���\>B�\?���C�j=                                    Bx���(  �          A (�@���z�?�33A ��C��R@���=q>#�
?fffC��3                                    Bx����  �          A ��@{��z�@:�HA���C��@{��?�z�A(�C��=                                    Bx���t  T          A ��@tz��G�@9��A�(�C��3@tz��ff?��A=qC�U�                                    Bx���  
�          A ��@l(����@K�A���C�b�@l(���\?�
=A0��C��)                                    Bx��	�  
�          A�
@a��G�@@��A���C���@a���R?�G�A"ffC���                                    Bx��f  
�          A\)@����p�@i��A�C��=@����Q�@\)Aj=qC�Q�                                    Bx��'  
�          A@�Q���H@[�A�
=C�� @�Q��	�@��AT��C�4{                                    Bx��5�  �          A=q@�Q��
�\@(Q�Av=qC�n@�Q��
=?�z�A�HC��                                    Bx��DX  "          A�@�ff���\@�Q�BP�\C���@�ff���@�(�B9�HC���                                    Bx��R�  "          A
=@���Y��A\)B_\)C�+�@����  @�p�BJ\)C�8R                                    Bx��a�  
�          A Q�@�(��tz�A (�BT��C�=q@�(���(�@��B?�C��                                     Bx��pJ  T          A!�@�Q���Q�@�BOC�N@�Q����@��B9  C�/\                                    Bx��~�  �          A!G�@�����@�z�BEQ�C��\@������@���B-  C�:�                                    Bx����  �          A ��@��H��@���BJffC�g�@��H��{@��B2�C��f                                    Bx���<  
�          A ��@��
����@��BG�C�9�@��
����@�ffB/33C���                                    Bx����  
�          A!�@������@�BBQ�C��{@����\@�G�B)�C���                                    Bx����  �          A"=q@�����@�\)BF�RC��@�����@�Q�B.�
C�z�                                    Bx���.  
�          A!��@�33��  A ��BS�HC��@�33����@�p�B={C��\                                    Bx����  
Z          A ��@����G�@�ffBQ{C��@�����@陚B:p�C��
                                    Bx���z  
�          A Q�@�33���@���BP�C�q@�33���
@���B:ffC��{                                    Bx���   "          A!�@��H���
@��\BL
=C�9�@��H���
@�B5�
C�Ff                                    Bx���  �          A!�@��R��@�G�BI
=C�^�@��R��@�(�B3
=C�|)                                    Bx��l  "          A"ff@���33@�(�BKC��@���33@�\)B5�HC��
                                    Bx��   �          A ��@�33�Z=qA�B\�C���@�33��
=@��RBI  C���                                    Bx��.�  
�          A!�@��R�=p�A\)Bb��C���@��R��G�@��BP�C��\                                    Bx��=^  �          A"{@�(��2�\A��Bf�
C�w
@�(��x��A�BUG�C���                                    Bx��L  T          A"ff@��\���A	G�Bf��C��=@��\�c33A
=BW(�C��{                                    Bx��Z�  "          A"ff@�z��,��AG�Bs
=C�j=@�z��u�AffB`�C���                                    Bx��iP  T          A"{@��$z�ABt(�C�33@��l��A33BbQ�C�<)                                    Bx��w�  �          A!�@�z��ffA�
Bz�RC���@�z��P  A
=qBj��C��                                    Bx����  
�          A ��@y����G�A{B��RC�k�@y���+�ABwz�C���                                    Bx���B  "          A"=q@vff�У�A�B�B�C�~�@vff�3�
A
=Bw�
C��                                    Bx����  
�          A#�@|(���33A\)B�33C�=q@|(��$z�A\)By=qC�u�                                    Bx����  	�          A#\)@vff>�A33B���?��H@vff��=qA=qB��HC�.                                    Bx���4  �          A"�H@g
=>��A(�B�Q�@��@g
=�uA�B��C��3                                    Bx����  
�          A"�R@��ͿL��A  B��C���@��Ϳ�(�A�B|��C�L�                                    Bx��ހ  "          A"�H@������A33B��qC�R@���{A\)Bw\)C�z�                                    Bx���&  
�          A"�\@aG��uAQ�B�k�C���@aG���G�A�HB�ǮC�'�                                    Bx����  T          A#33@c33    A��B�=qC���@c33���HA�B��
C���                                    Bx��
r  "          A$  @u����
A(�B�#�C��
@u����HA�HB��HC�7
                                    Bx��  T          A$Q�@���>k�A�HB�ff@L��@����s33A{B��C�e                                    Bx��'�  T          A$(�@mp��.{A��B��\C���@mp���{A\)B��3C��                                    Bx��6d  �          A$��@X�þB�\A�B�z�C�Z�@X�ÿ�33A{B�B�C�˅                                    Bx��E
  �          A%�@(����A33B�#�C�h�@(��G�A  B�G�C��                                    Bx��S�  
�          A%�@AG���ffA�B��C�` @AG��/\)A\)B�ffC��                                    Bx��bV  "          A%G�@��H��\)A�B
=C�4{@��H�   A  Bs��C�
=                                    Bx��p�  "          A%�@s33�7
=Ap�By�\C�}q@s33�|��A
�RBf�HC���                                    Bx���  �          A$(�@����8Q�A33Bt(�C�9�@����|��A��Bb�C�˅                                    Bx���H  
�          A$��@w
=�%A�\B|z�C�\@w
=�k�Az�Bj��C�*=                                    Bx����  �          A&=q@�\)��G�A33Byp�C�C�@�\)�7
=A�RBlC���                                    Bx����  �          A&=q@��\��z�A�HBx=qC��@��\�0  A�\BlG�C��                                     Bx���:  
�          A%G�@�33��
=A=qBz33C�˅@�33�G�A
=Bp�\C�޸                                    Bx����  "          A$��@qG��Mp�A��Bu33C��@qG���Q�A	p�Ba�C��q                                    Bx��׆  "          A"ff@j�H�s�
A	�Bi��C��
@j�H���A��BU
=C���                                    Bx���,  T          A"=q@L(���Q�A��B^�C���@L(���
=@�\)BF�HC��R                                    Bx����  
�          A"�\@g
=����A\)Bb�C�{@g
=���@���BL�C�B�                                    Bx��x  
�          A!�@p���w
=A��Bg�C�!H@p�����HA ��BR�RC��                                    Bx��  "          A#�@c�
��(�A�RB`�C��{@c�
���\@�33BK
=C��H                                    Bx�� �  
�          A$z�@p���fffAp�Bn{C�q@p�����ABZ\)C���                                    Bx��/j  �          A$  @O\)��=qA33BhC�p�@O\)����A=qBR��C��R                                    Bx��>  "          A$(�@S�
����A�Bh��C��@S�
��Q�A�RBS
=C�q                                    Bx��L�  "          A%�@Z=q�dz�A��Buz�C�Ф@Z=q���HA	�Ba33C�J=                                    Bx��[\  "          A#\)@U��2�\A�
B���C�H@U��u�A��Bp�RC�|)                                    Bx��j  �          A#
=@fff�P��AffBu��C��f@fff��Q�A�Bb�C��                                    Bx��x�  �          A#�
@aG��z�HAQ�Bl  C��@aG�����AQ�BW��C�ٚ                                    Bx���N  �          A$Q�@aG��u�A��BnG�C�J=@aG����ABZ(�C��                                    Bx����  "          A$��@r�\��\)A	G�Bb  C���@r�\��p�A ��BM�HC�q                                    Bx����  "          A$��@l(���\)A	�Bc�C���@l(���p�Ap�BOp�C��H                                    Bx���@  
�          A$(�@g
=�{�A(�Bj�C�G�@g
=��(�AQ�BV�
C�9�                                    Bx����  T          A$��@p����RA=qBz{C��R@p���ffA	�Bb��C���                                    Bx��Ќ  �          A%G�@ ������A(�Bx��C��{@ ����Q�A(�Bb�\C�Q�                                    Bx���2  
�          A%��@�p���(�A(�B~33C�Z�@�p��>�RA�
BqQ�C�9�                                    Bx����  �          A%@�z��0��A
�HBdz�C�L�@�z��l(�AG�BV��C�y�                                    Bx���~  T          A&{@�{�<(�A�
Bfz�C��
@�{�xQ�ABWC�9�                                    Bx��$  �          A'
=@��H�ǮA��Bs(�C�
@��H�"�\A�Biz�C�.                                    Bx���  �          A&ff@�33���A��B�33C��R@�33�z�AffBv�HC��R                                    Bx��(p  �          A$��@H���J=qAQ�B���C�ff@H����z�A{Bn�C���                                    Bx��7  "          A%��@QG���p�A�Bg{C�>�@QG����HA\)BR�C���                                   Bx��E�  �          A%�@c33��p�AG�BW�C���@c33����@�Q�BCQ�C���                                   Bx��Tb  �          A&{@g���Q�A�BUz�C��@g����
@��BA=qC��
                                    Bx��c  "          A&{@P�����
AG�BL�C��@P����ff@�{B7\)C�n                                    Bx��q�  
(          A&{@U���z�@�B>�
C�=q@U����@߮B)\)C��q                                    Bx���T  
(          A%�@g�����@�  B0��C�� @g���  @У�B\)C�C�                                    Bx����  �          A$��@c33����@��
BH�RC�^�@c33�ə�@�  B4=qC��                                    Bx����  �          A%�@5���G�@�p�B@
=C��@5��ᙚ@�
=B*(�C��                                    Bx���F  T          A$Q�@l(���@ȣ�B�C�o\@l(��陚@���BC�ff                                    Bx����  �          A"{?���θR@�\)B@{C��?����{@���B)p�C�T{                                    Bx��ɒ  �          A   @33���@�  Bp�C��@33���
@�\)B
=C�#�                                    Bx���8  T          A�@5��  @n{A�Q�C��3@5��p�@4z�A��C���                                    Bx����  �          A!p�@(����\)@��B��C�&f@(����
@��A�G�C��H                                    Bx����  "          A ��?h���
=@���B	�C�,�?h����@���A噚C��)                                    Bx��*  �          A Q�?�p���\)@�p�B7�C�4{?�p����@�
=B!p�C���                                    Bx���  �          A!G�@�
��@�Q�BJ��C��H@�
��p�@�(�B5=qC��=                                    Bx��!v  
�          A$  @K�����@�ffBKG�C���@K�����@�B7G�C�8R                                    Bx��0  
�          A#�@>�R��{A�BRz�C�` @>�R�ƸR@�B>p�C���                                    Bx��>�  T          A#33@Mp����\@��BKffC���@Mp��ʏ\@�\B7�RC�u�                                    Bx��Mh  
�          A!@5���@��BJ�\C�+�@5��
=@�
=B6\)C��)                                    Bx��\  T          A!�@/\)��@�
=BG
=C�k�@/\)��z�@�B2C�8R                                    Bx��j�  T          A"{@<(��ƸR@�{B=33C��\@<(���z�@�=qB)(�C��\                                    Bx��yZ  
�          A"=q@QG���33@��B;�HC�  @QG��أ�@ٙ�B(ffC��f                                    Bx���   
�          A!@R�\���@�Q�B@
=C���@R�\���H@�p�B,�
C�@                                     Bx����  "          A ��@Fff���@�  BAffC���@Fff���H@��B.
=C��{                                    Bx���L  �          A!p�@�Q����@�RB7Q�C�t{@�Q���z�@�z�B%ffC��                                    Bx����  T          A!��@�=q���
@θRB (�C��@�=q��{@��HB=qC��\                                    Bx��  �          A!p�@�  ���@�ffB/Q�C���@�  ��
=@�(�B�
C��f                                    Bx���>  �          A (�@������@���B\)C��q@�����
=@��RA�C��                                    Bx����  �          A!��@�33���
@�
=B
ffC�=q@�33���
@�z�A�\)C�!H                                    Bx���  
�          A!�@�G�����@�33B�C�g�@�G���p�@�  B��C�T{                                    Bx���0  �          A"{@�����z�@��B�C�y�@�����@���B
=C�
                                    Bx���  
Z          A#\)@�\)���@�Q�BC�/\@�\)��p�@�\)B�RC��3                                    Bx��|  "          A"=q@�  ��G�@�\)B�RC�c�@�  ��33@��RB�HC��                                    Bx��)"  �          A!��@�  ��(�@�33B�\C�˅@�  ����@��\B
=C��=                                    Bx��7�  �          A!@�(����
@��RB�
C��=@�(����
@���Bz�C��\                                    Bx��Fn  
�          A!�@�\)����@�(�B{C��R@�\)��z�@���A�33C��f                                    Bx��U  �          A!�@�p��Ӆ@��B��C���@�p���33@�z�B�C���                                    Bx��c�  �          A z�@����  @�z�B  C�Ф@����\)@�G�A���C��=                                    Bx��r`  �          A z�@���ҏ\@��B  C���@�����@��\B �
C��
                                    Bx���  �          A z�@�Q����
@��HBC�b�@�Q����
@���B��C�`                                     Bx����  
�          A Q�@��
��p�@�\)B!33C��@��
��ff@�\)B�C�                                    Bx���R            A�@�G���  @��B
=qC�33@�G���{@�  A�33C�O\                                    Bx����  n          A��@�\)���@�p�A��C�b�@�\)��
=@��HA�{C���                                    Bx����  �          A��@����G�@�=qA��C�.@�����@�
=A�z�C�~�                                    Bx���D  "          A��@�����H@��\A��
C�c�@����R@�  A�p�C���                                    Bx����  
�          A(�@�z��ۅ@��A�Q�C�j=@�z���
=@��HA�(�C���                                    Bx���  T          A(�@�z���p�@��A�
=C�L�@�z����@\)A�
=C���                                    Bx���6  
�          A  @�
=��\@�  A�{C���@�
=��p�@z�HA�C���                                    Bx���  
�          A
=@�����
@�  A뙚C��@����\)@�p�Ȁ\C��                                     Bx���  
�          AG�@�Q��
{@��Af�\C�q�@�Q����?�G�A$��C�<)                                    Bx��"(  T          A�H@����33>\)?L��C���@�����H�\)�O\)C��                                    Bx��0�  
�          A ��@g�����G�����C��q@g��\)��p���\C���                                    Bx��?t  T          A ��@aG��=q���Q�C�B�@aG�����ff���C�U�                                    Bx��N  
�          A!@^�R���W
=���HC�3@^�R�
=�p����=qC��                                    Bx��\�  
�          A#
=@y���ff�����
C�Ff@y���p���\)��=qC�U�                                    Bx��kf  "          A#\)@p������
=�C�Ф@p����\�������HC��                                     Bx��z  
�          A#�@z=q�
=���
���
C�B�@z=q��\�8Q�����C�J=                                    Bx����  
�          A#33@dz���׽�\)��p�C�C�@dz��(��E����HC�L�                                    Bx���X  T          A#33@h���(�������C�y�@h���\)������ffC���                                    Bx����  
�          A#
=@X���G����@  C���@X���Q쿝p���p�C��                                    Bx����  
�          A33@��H���?�@S33C�@��H�녽�\)���C��                                    Bx���J  
�          A��@��H�	�?&ff@u�C��@��H�	p�=#�
>k�C��                                    Bx����  
�          A
=@|����>���?�z�C�Ǯ@|�������R��ffC�Ǯ                                    Bx����  �          Az�@�=q��?�ffA{C��{@�=q��\?z�H@���C��\                                    Bx���<  �          A�H@����G�?0��@�Q�C�=q@����=�G�?&ffC�1�                                    Bx����  �          A Q�@��
��H?(��@r�\C�@ @��
�33=��
>�C�5�                                    Bx���  �          A!�@�����=q��ffC�H@���Q�ٙ��Q�C�q                                    Bx��.  T          A#33@~�R�  �У��
=C���@~�R���  �K�
C���                                    Bx��)�  "          A"�H@\(���H����z�C�@\(�����G��N{C�&f                                    Bx��8z  �          A#�@a��
=���"�HC�@ @a�������[�C�e                                    Bx��G   :          A$��@s�
�\)���H��\C���@s�
�p����:ffC��                                    Bx��U�  �          A%�@{��ff��z��z�C�U�@{��Q��G��K\)C�y�                                    Bx��dl  �          A%�@�=q��\��  �G�C��3@�=q����
=�;\)C��{                                    Bx��s  �          A%@�G���H��  �p�C���@�G�����ff�;
=C��)                                    Bx����  "          A&ff@|���(������   C�Ff@|���ff��\�5G�C�e                                    Bx���^  �          A&�\@u�G������C��=@u����z��(��C�                                    Bx���  "          A&ff@c�
����=q��ffC�\@c�
�{��
=�Q�C�%                                    Bx����  
�          A'\)@E�����4  C��{@E�G��'��iG�C��3                                    Bx���P  �          A'33@#33�G��Q���ffC�s3@#33���vff��33C��q                                    Bx����  
�          A'\)@@  �{�-p��pz�C���@@  �\)�Q���ffC��                                    Bx��ٜ  
�          A'�@@  ����G
=��{C���@@  �p��j=q���C���                                    Bx���B  �          A'�@B�\��H�'
=�f�HC���@B�\�(��J�H��
=C��                                     Bx����  "          A'33@S33�33��(��.=qC�e@S33���!��`z�C���                                    Bx���  �          A'\)@?\)�\)�p��Y��C��=@?\)����@�����
C���                                    Bx��4  
�          A(��@7
=�33�?\)���C�:�@7
=�(��a���z�C�aH                                    Bx��"�  "          A(��@&ff��R�Vff��(�C���@&ff�\)�x������C��\                                    Bx��1�  T          A(��@ff�33�\������C�˅@ff�  �~�R��33C��\                                    Bx��@&  
�          A(��@*�H�
=�N{��{C���@*�H�  �o\)��(�C���                                    Bx��N�  "          A(��@P  ����>�R����C�e@P  �{�_\)����C��                                    Bx��]r  "          A%��@4z���\�u���=qC��\@4z��
=����ŅC��                                     Bx��l  �          A$(�@\)�������\)C���@\)�	��\)��\C��                                    Bx��z�  
�          A$Q�@ff����ff��{C�s3@ff�����������C��=                                    Bx���d  T          A#\)?�p��
=q���\���C�w
?�p���������RC��=                                    Bx���
  �          A"ff@ff����������C��H@ff�\)��
=���C��q                                    Bx����  �          A"�H@�������33���C��{@�����������{C��q                                    Bx���V  "          A"�H@��������G�C��{@��(������C��q                                    Bx����  �          A"�\@����������C�AH@��	p������\)C�o\                                    Bx��Ң  "          A"�\@.{�ff������
=C���@.{�ff��p�� �C�<)                                    Bx���H  T          A#�
@(���33����  C�t{@(�������H��\C���                                    Bx����  �          A$  @.{�  ������
C��f@.{�Q���Q���{C��)                                    Bx����  
�          A$  @.�R�z���=q��p�C���@.�R�����
=��p�C��R                                    Bx��:  "          A#�
@W��
=q���H�ȣ�C��f@W���H��
=���
C��H                                    Bx���  �          A$  @0  �
�\��  �ܸRC�Ф@0  �
=��(���{C��                                    Bx��*�            A$��@
=�
ff��p����C�ٚ@
=��\��G��G�C��                                    Bx��9,  ;          A$  ?�\)��
�����\)C�>�?�\)�(���\)�\)C�b�                                    Bx��G�  
�          A#�?�Q��  ��33��33C��=?�Q��z����R�(�C���                                    Bx��Vx  
�          A$��?�33�����ff���C�ff?�33�����=q��  C��                                     Bx��e  
�          A%�?��
�{��(����HC�N?��
�
�\���� C�c�                                    Bx��s�  
�          A$��?L���������G�C��H?L���(���  ���C��3                                    Bx���j  
�          A%p�?#�
�  ���� �C�3?#�
�Q����\�	�C�!H                                    Bx���  
�          A&�\?L���	������\C���?L������(��ffC��
                                    Bx����  �          A&�H?��Q���\)��C��=?��z��ə���
C��
                                    Bx���\  �          A&�H>��H����ff�  C���>��H�{��Q���\C��R                                    Bx���  
�          A'
=>�{����ʏ\�\)C�/\>�{� ����(����C�8R                                    Bx��˨  "          A&�H<��
�ff��Q��z�C�{<��
����ٙ��"C�{                                    Bx���N  "          A&=q>#�
��
=��{�'�RC���>#�
��
=��R�/�
C���                                    Bx����  T          A%<#�
�����߮�)�C�<#�
������  �1��C�                                    Bx����  
�          A$��>�z����������C��>�z���33��p��!\)C�3                                    Bx��@  
�          A%�?��R�G���z����C�>�?��R�{����Q�C�Y�                                    Bx���  
�          A%?�����p���
C���?����\��{�33C���                                    Bx��#�  
�          A&=q?B�\�  ��
=���C��?B�\� ����\)��C��
                                    Bx��22  T          A&�R?�����=q�߮�)p�C�� ?�����33��
=�0z�C���                                    Bx��@�  �          A&{?�=q��\)��  �33C��q?�=q�����׮�"{C��)                                    Bx��O~  �          A&ff?�G���(���33��RC�n?�G����ڏ\�$p�C���                                    Bx��^$  "          A(Q�?˅�G��ƸR��HC�g�?˅�ff��ff�p�C��f                                    Bx��l�  "          A(  ?fff���H�����$�\C�C�?fff��z����
�+
=C�Y�                                    Bx��{p  �          A&�H?����  ��{�'
=C��?����������-ffC��                                    Bx���  "          A((�?^�R��33��p��$��C�(�?^�R�����(��*��C�=q                                    Bx����  �          A'�?����Q���{�5�C�t{?�������(��;z�C��3                                    Bx���b  �          A'33?\(��陚��(��4�\C�Y�?\(�����=q�:\)C�o\                                    Bx���  T          A&�H?p������\)�?\)C��R?p���׮�����E  C��3                                    Bx��Į  "          A&=q?J=q��  ��z��5z�C�q?J=q��=q����:��C�1�                                    Bx���T  
�          A%�?�����33��{�/�RC�{?��������5
=C�33                                    Bx����  T          A&ff?��������ᙚ�*��C��=?������
��R�0{C���                                    Bx���  
�          A'\)?���� (���33��RC�AH?�����������!�RC�U�                                    Bx���F  �          A&�H?z�H��Q������%�RC��R?z�H����=q�*�\C��=                                    Bx���  	          A(  ?�p���ff����(  C��?�p������,�C���                                    Bx���  �          A%?�=q������  �"�
C���?�=q��z�����'\)C��                                    Bx��+8  
s          A!�>�{������=q�QQ�C��R>�{��(����U�RC���                                    Bx��9�  
�          A ��?��������B(�C�Ǯ?����ə����FG�C��f                                    Bx��H�  "          A"�\@�����z��LffC��R@���������P(�C���                                    Bx��W*  T          A#
=?��������z��d��C�L�?������
�	��h�\C��                                    Bx��e�  �          A"�R?^�R��{�p��q��C�H?^�R��G���R�uz�C�&f                                    Bx��tv  �          A�
?E������	p��m\)C�Z�?E������
�R�p�HC�y�                                    Bx���  �          A (�>������c�C�&f>������
=�g{C�33                                    Bx����  �          A33?+������ ���X\)C�O\?+���p����[��C�`                                     Bx���h  
�          A�\?^�R��������S�HC�,�?^�R��G�����V��C�@                                     Bx���  �          A�R�u��(��=q�]G�C����u�������`33C�                                    Bx����  
�          A�R��Q���p��(��b�RC�
��Q���=q�G��ep�C�                                    Bx���Z  �          A�������
=�(��vffCz��������
����x�
CzE                                    Bx���   T          A Q��Q���33���w\)CxǮ��Q���Q��Q��y��Cx\)                                    Bx���  �          A�ÿG���G����F{C�y��G���
=���HQ�C�n                                    Bx���L  
�          A
=��=q������G��:\)C��3��=q���H��33�<p�C��\                                    Bx���  
�          A�>u��33���
�+�
C��
>u�ᙚ���-C���                                    Bx���  "          A
=�=p������ff�R�HC�b��=p���Q�����T��C�XR                                    Bx��$>  "          Az�Q��У����
�9
=C�ff�Q���
=��p��:��C�`                                     Bx��2�  �          A  ��=q������H�J�C�Ф��=q�����(��LQ�C��f                                    Bx��A�  
�          Az��\����  �[33C{{��\��(������\ffCz�                                    Bx��P0  T          A����=q�������^33C|����=q������R�_G�C|ff                                    Bx��^�  �          A��Dz��w
=��{�h�\Cg���Dz��u���ff�iG�CgJ=                                    Bx��m|  �          A33�h���w������^�
Cb�=�h���vff����_ffCb��                                    Bx��|"  �          A�\�j=q�}p����
�Z�HCc:��j=q�|(���(��[Q�Cc{                                    Bx����  m          A���e���
��  �WCd�3�e��33��Q��X{Cdٚ                                    Bx���n  �          A(��qG���������Q�Cc�qG���������R�Cc�3                                    Bx���  �          Az���
=�333�����c��CU���
=�2�\�����c�RCU}q                                    Bx����  �          A����(��8����G��XffCS�R��(��8�������X\)CS�q                                    Bx���`  �          A�������(���33�ZCN�����������H�Z��CO                                    Bx���  T          A���ff�����H�V�CK����ff�(���\�V\)CKٚ                                    Bx���  
�          A��\)�������Y�RCN�f��\)��H�����Yz�CO{                                    Bx���R  "          Aff����{��G��]��CPs3����\)�����]��CP�                                    Bx����  �          A����H�4z���33�V��CS�\���H�6ff���H�Vz�CS��                                    Bx���  �          A����=q�dz���Q��Iz�CZ���=q�g
=�߮�H��CZB�                                    Bx��D  T          A�
���������(��-{C`s3�������H�Å�,33C`��                                    Bx��+�  �          A��\)�������wCE���\)��=q��33�wQ�CF��                                    Bx��:�  T          AQ���G��=p����H�Z�\CV����G��@����=q�Y�CW�                                    Bx��I6  �          A  ��G��mp���(��Q�C^����G��p����33�P�C^��                                    Bx��W�  �          A33�x���vff��=q�Q{C`���x���z=q��G��OCa#�                                    Bx��f�  �          A33�c33��=q��z��Sp�Cd���c33��z���33�Q�HCeY�                                    Bx��u(  T          A33�[��n{��(��^=qCcY��[��s33��33�\��Cc��                                    Bx����  "          A(��.{���H��(��]ClO\�.{������H�[Cl�=                                    Bx���t  
Z          A�\�(������ə��.��C�!H�(����(���
=�,=qC�(�                                    Bx���  T          A��?��\��{��z��33C�33?��\��  ������C�(�                                    Bx����  �          A��@8Q��33�A����
C���@8Q����;���Q�C���                                    Bx���f  
�          AQ�@=q�
ff�w
=��  C�˅@=q�33�o\)��{C��H                                    Bx���  �          A"ff@7����Q����\C���@7���R�x����ffC��                                    Bx��۲  �          A$Q�@Dz��z����H��=qC���@Dz��p����R��C���                                    Bx���X  
�          A#�
@)���(��^�R��
=C���@)������U��(�C��                                    Bx����  "          A"�H@,���ff�e���ffC�9�@,���\)�[���33C�,�                                    Bx���  T          A ��@5��
=�4z����HC���@5���
�*�H�v�HC��                                     Bx��J  �          A�@p���tz���=qC��@p���H�j=q��ffC��
                                    Bx��$�  �          A=q@*=q����������C���@*=q�������ʸRC���                                    Bx��3�  �          AQ�@33���AG���{C�c�@33�Q��6ff����C�XR                                    Bx��B<  �          A�
@�H����"�\�x��C��=@�H��
=�f�HC��                                     Bx��P�  �          AG�@(Q��G��\�ffC��@(Q����=q� Q�C�\                                    Bx��_�  �          A�@   �p�����  C��3@   �녿����C���                                    Bx��n.  T          AG�@@���G�>k�?�{C�,�@@����>�(�@#�
C�.                                    Bx��|�  T          AQ�@HQ����W
=���
C���@HQ����#�
�L��C��q                                    Bx���z  �          A�\@\(��33?c�
@�C��=@\(��
�H?��@�C���                                    Bx���   "          A�R@\)���
@��A��C��@\)��Q�@���A�RC�AH                                    Bx����  �          A(�@��H��=q@�p�B=qC�}q@��H��@��HB	(�C��q                                    Bx���l  �          AQ�@n{��R@�\)A�ffC���@n{��\@�p�Bp�C��                                     Bx���  �          A(�@W
=��=q@�(�A�\)C���@W
=��ff@�=qA�ffC�&f                                    Bx��Ը  �          A(�@c33��\@�Q�A�\C���@c33��R@�
=A��
C��
                                    Bx���^  m          A�@��\��G�@��A�C���@��\����@�Q�A�z�C�                                    Bx���  T          A��@�p���\@�ffA�ffC�<)@�p���{@���B�
C�~�                                    Bx�� �  �          A�@�  ���H@���A�p�C��{@�  ��{@�\)B�\C��R                                    Bx��P  "          A�
@tz���G�@���B z�C�<)@tz���z�@��B�C��H                                    Bx���  "          A  @����ff@w�A��C�.@����\@�33A�G�C�b�                                    Bx��,�  �          A�@����H@ffAd(�C��@���Q�@&ffA|��C�1�                                    Bx��;B  �          A\)@�  ���?���A1p�C�\)@�  ��p�@�AJ�HC�u�                                    Bx��I�  
Z          A�H@���� ��?k�@��C�Y�@���� Q�?�Q�@�C�g�                                    Bx��X�  �          A@�����G�?�@�C�c�@�����  ?�
=A��C�u�                                    Bx��g4  T          A{@����G�?��
@ʏ\C��3@���� ��?��A ��C��                                    Bx��u�  
�          A{@�z���\?�Q�A?�C��@�z���  @p�AZ{C�7
                                    Bx����  T          A��@�
=���
@  A_�C���@�
=����@ ��AzffC�Ф                                    Bx���&  �          A�@���p�@=qAo�
C���@���=q@*�HA�G�C���                                    Bx����  �          A��@�ff���H@�AW�
C�e@�ff��  @(�ArffC���                                    Bx���r  "          A�@��\���H@ffAi�C�
@��\�߮@'�A��RC�G�                                    Bx���  "          A�@�Q���  @(�AXQ�C���@�Q����@{At��C�Ǯ                                    Bx��;  �          A@�=q����?�\)A33C��{@�=q��\?�33A;\)C��
                                    Bx���d  �          A�
@��\��
=����p�C�  @��\��\)��G��#�
C���                                    Bx���
  �          A�
@�=q��R�
=�c33C��)@�=q��\)��\)�ٙ�C��3                                    Bx����  �          A�H@�\)��p��
=�g�C���@�\)����\)��p�C���                                    Bx��V  �          A�@�{��R?�Q�AD  C��f@�{���
@��AdQ�C��                                    Bx���  �          AG�@�=q��33@tz�A��HC�c�@�=q��@�(�A�p�C��
                                    Bx��%�  �          AG�@|�����@���Bz�C�W
@|����@�B{C�˅                                    Bx��4H  �          A��@Y�����
@��B33C��@Y����33@�{B&33C���                                    Bx��B�  �          A�@Z�H��G�@�{B�\C�Ф@Z�H����@�\)B�RC�N                                    Bx��Q�  �          A��@����Q�@��A�=qC�� @���ᙚ@�=qA��C��R                                    Bx��`:  �          A@�33��(�@�\Ac�
C�u�@�33��Q�@)��A��RC���                                    Bx��n�  T          A�@[�����Ǯ�{C�+�@[��<#�
=L��C�'�                                    Bx��}�  �          A��@HQ���׿�33�(Q�C�\)@HQ�����R��{C�G�                                    Bx���,  �          Aff@���=q�&ff��(�C�\)@���(�����ep�C�@                                     Bx����  T          A\)@0����H�ff�s\)C�W
@0���z��Q��H(�C�:�                                    Bx���x  �          A�
@����{��=q�ٙ�C���@����ff>�?Y��C�Ф                                    Bx���  �          A33@��
��p����R��(�C�@��
��p�=���?�RC��                                     Bx����  �          A�R@��
����Tz����\C�Ф@��
���;�
=�,��C��                                    Bx���j  �          A  @7��޸R�����HC�7
@7���ff�����\C��H                                    Bx���  �          A=q@E��G��\)�У�C�#�@E����e���  C��H                                    Bx���  �          A�
@333��(���G���{C�c�@333���
��(����HC�{                                    Bx��\  �          A��@�R��G����\�p�C�
=@�R��33��ff���C���                                    Bx��  
�          A(�@(����
=��\)�C��{@(����Q���33��C�K�                                    Bx���  �          A\)@=q��\)�����߮C��f@=q���R�w
=���C�j=                                    Bx��-N  m          A=q@+���\)��\)�뙚C��q@+���
=������
=C��3                                    Bx��;�  �          AQ�@"�\���H��z��G�C��)@"�\��z���
=���C�}q                                    Bx��J�  "          A\)?��������\)�<�C���?�������˅�/z�C��f                                    Bx��Y@  "          AQ�@
=������=q���C�}q@
=��\)������C�                                      Bx��g�  
�          A33@G��׮��G��C�Z�@G���\��(����C���                                    Bx��v�  T          A{?������Q��$��C��?���Q���33�\)C��3                                    Bx���2  "          A��@Q�������
�C�t{@Q���\���\)C�)                                    Bx����  �          A��?�33��G���33�{C���?�33��33�����ffC�@                                     Bx���~  �          A��@z���z���\)�G�C�]q@z���
=��G���C�                                      Bx���$  T          A�?�\)��Q���  �&Q�C�H?�\)���
���\�\)C���                                    Bx����  �          A��?�����Q��׮�?{C�.?��������H�0�C�ٚ                                    Bx���p  
Z          A?��H������ff�=�C��{?��H��{��G��/\)C�W
                                    Bx���  �          A(�?s33�����\�Y�\C�=q?s33��(���
=�J�
C�Ф                                    Bx���  T          A33?aG���(���p��8=qC��?aG���G���Q��)G�C�Ф                                    Bx���b  "          A=q?�\��\)��  �\��C�E?�\��ff��p��N33C�g�                                    Bx��	  
�          Az�@�
��\)��Q���\C�w
@�
��������z�C���                                    Bx���  �          A��@r�\��\@
=qAjffC���@r�\��p�@)��A��HC���                                    Bx��&T  T          A	G�@aG���Q�?�p�AAG�C��@aG���(�@{Az{C�'�                                    Bx��4�  "          Aff@��������\�pQ�C�R@��������G��8  C��                                     Bx��C�  "          A�@r�\��ÿ�  ��C��@r�\�
ff�c�
��z�C��{                                    Bx��RF  �          A��@n�R�(���
�H  C��{@n�R�
{�����Q�C��                                    Bx��`�  T          AQ�@Tz����@����  C�Ф@Tz�����=q�j�HC��
                                    Bx��o�  T          Aff@/\)�p��w
=�£�C�^�@/\)����QG���33C��                                    Bx��~8  �          A(�@*=q��������  C��@*=q���
��\)��z�C��                                     Bx����  �          A?�z������������C��H?�z���{�����\)C��                                     Bx����  �          A��?����ff��=q���C���?�����H��G��
=C�W
                                    Bx���*  �          A
=@   ��R�Mp���G�C��q@   �=q�%��C���                                    Bx����  
�          A33@W
=�=q������\C���@W
=�\)�\)�c33C��R                                    Bx���v  
�          A
=@X���z�>��@<��C��R@X����?�\)@��C��                                    Bx���  �          A
=@>�R�
�H>�{@��C�xR@>�R�
{?�G�@ʏ\C��                                    Bx����  T          A�
@P  �	�?���@��C�e@P  �\)?��A3
=C��                                    Bx���h  �          A�H@Vff��?G�@�z�C���@Vff�=q?�Q�A�C��                                    Bx��  m          A{@x������@(�Ad��C�t{@x����R@2�\A���C�                                    Bx���  �          A\)@o\)���?�  A�HC��=@o\)���H@��A]G�C���                                    Bx��Z  �          A�@@���=q?Y��@��RC��H@@�����?\A��C��)                                    Bx��.   "          A  @X���{?��A7�
C�T{@X�����R@��A|z�C���                                    Bx��<�  �          A��@i�����?�p�A0(�C�!H@i����@Q�Atz�C�\)                                    Bx��KL  ;          Ap�@aG���
?��Ap�C��@aG��?��RAK
=C���                                    Bx��Y�  �          AG�@g
=���@  ��G�C�ٚ@g
=�Q콏\)��
=C��                                    Bx��h�  �          A��@W���Q�?�33@�
=C��R@W���z�?�ffAB{C���                                    Bx��w>  �          AQ�@Z�H��33@|��A��C�y�@Z�H�߮@��A���C��                                    Bx����  
�          A�@�(���{@aG�A�C�#�@�(��Ӆ@�33Aڣ�C���                                    Bx����  �          AQ�@q���{@3�
A�\)C��@q���p�@]p�A���C�y�                                    Bx���0  
�          A��@S�
�G�@p�Ac�C�%@S�
���@9��A��\C�n                                    Bx����  �          A{@�R��\@�A~�\C�u�@�R���@HQ�A���C��{                                    Bx���|  �          A��@7
=�陚@���A�=qC���@7
=��z�@���B�
C�E                                    Bx���"  �          AQ�@XQ����@�z�B  C�{@XQ����@�B%C�
                                    Bx����  �          A(�@.{��=q@�Q�B9��C�  @.{���R@�\)BK�HC�W
                                    Bx���n  �          A(�@�
��ff@���B�\C���@�
��p�@�(�B-=qC�7
                                    Bx���  �          A�@���޸R@�G�B�
C��@����ff@���B"ffC�5�                                    Bx��	�  �          A��@S33�ҏ\@��\B�C�U�@S33��=q@��B"�C�J=                                    Bx��`  T          A=q@n{��Q�@��HB G�C�e@n{��G�@�{B��C�J=                                    Bx��'  �          A�R@qG���(�@�A���C�` @qG����@���B�HC�<)                                    Bx��5�  �          Aff@��
����@��Bz�C�l�@��
��G�@��B��C��=                                    Bx��DR  �          A�\@���Q�@�Q�A�C���@���G�@���BQ�C��3                                    Bx��R�  T          A��@^�R��@w�A�33C��f@^�R�޸R@���A�=qC�G�                                    Bx��a�  �          A��@dz����@p��Aģ�C��)@dz���Q�@�ffA��
C�z�                                    Bx��pD  T          A�@,(���  @�=qA�Q�C�� @,(���=q@���B �
C�k�                                    Bx��~�  �          A  @J�H��Q�@s�
Aȏ\C�o\@J�H��@���A�G�C��                                    Bx��  
Z          A�
@J=q��ff@UA���C�(�@J=q���H@�=qA��
C��f                                    Bx�6  �          A�\@�33����@��\B${C��)@�33��ff@�=qB5p�C���                                    Bx�ª�  T          A@���y��@���B1G�C�u�@���R�\@���B>�
C��\                                    Bx�¹�  "          A  @������@ۅBP  C�o\@��׿�  @�=qBY(�C��=                                    Bx���(  ;          A	��@z�H��Q�@���B1=qC��q@z�H��(�@θRBBp�C��q                                    Bx����  m          A	G�@\(���z�@�{BDp�C��@\(��n{@�33BV  C�g�                                    Bx���t  T          A
ff@�\)�p  @�G�B<�C��@�\)�Fff@�z�BJ��C��                                    Bx���  �          A  @XQ������z��z�C�t{@XQ���
=?�@^�RC�xR                                    Bx���  ;          Ap�@#�
�=q�(�����C�y�@#�
��\>L��?��C�t{                                    Bx��f  "          A�@�  ���\@��B.�C�b�@�  �[�@љ�B<�HC�˅                                    Bx��   �          AQ�@�\)���H@�\)BG�C��R@�\)��  @��RB%�C�s3                                    Bx��.�  "          Az�@�
=���@�Q�A�
=C���@�
=��{@��B(�C���                                    Bx��=X  
�          AQ�@��R��z�?�{A$��C�%@��R��@��Ax  C��                                     Bx��K�  �          A  @�����@{A\)C�\)@����@O\)A�33C���                                    Bx��Z�  
�          A  @������@Mp�A�(�C�e@�����@}p�Aң�C�                                    Bx��iJ  �          A�H@�\)�ڏ\@aG�A�Q�C���@�\)���@�\)A�
=C�xR                                    Bx��w�  ;          A�\@�
=��=q@s�
A˙�C��q@�
=���
@���A�33C��                                     Bx�Æ�  m          A33@�=q���H?�  A�C�>�@�=q��(�@�At��C���                                    Bx�Õ<  �          A  @�����{>�Q�@z�C��f@�����33?�(�@��
C��f                                    Bx�ã�  T          A�
@�
=���\?��@�\C�*=@�
=���?��RAM��C�n                                    Bx�ò�  T          Az�@�\)��=q=�\)>�C��@�\)��Q�?n{@�{C��                                    Bx���.  
�          A�@�\)����>u?�G�C��@�\)��=q?�{@�33C�5�                                    Bx����  �          A�@qG��=�\)>�(�C�u�@qG�� ��?xQ�@�
=C��=                                    Bx���z  �          A=q@������H?�  A�C��
@�����(�@At��C��                                    Bx���   T          A�
@z�H��=q�����z�C�3@z�H���?�@g�C�R                                    Bx����  m          A��@qG�� �Ϳp������C���@qG�������
��C�}q                                    Bx��
l  m          A(�@z=q�G�?0��@�p�C��@z=q��ff?�{A$��C�{                                    Bx��  
�          A\)@QG���?�G�A��C���@QG�� z�@(�Ac
=C�3                                    Bx��'�  "          A
=@�����p�?z�@r�\C���@�����?��HA��C��3                                    Bx��6^  "          A�
@S�
��?�
=A=qC��@S�
� (�@�At��C�5�                                    Bx��E  T          A\)@W
=��
?5@���C��@W
=���?�z�A+33C�E                                    Bx��S�  ;          A\)@QG��=q?�=qA"�HC��@QG�����@!G�A��C�:�                                    Bx��bP  	          A33@�Q���?��@�Q�C�U�@�Q���?�(�AL��C���                                    Bx��p�  T          Az�@�G���G�>Ǯ@   C�  @�G���{?�  @�\)C�/\                                    Bx���  �          A  @������>��
@33C�
@�����
=?�z�@���C�C�                                    Bx�ĎB  "          A�
@���ָR?��@ٙ�C���@������?���A>�HC���                                    Bx�Ĝ�  �          A(�@���ff?z�H@ȣ�C��@�����?�  A3\)C�Ff                                    Bx�ī�  �          Az�@�=q��>Ǯ@\)C�aH@�=q�\?���@�RC��R                                    Bx�ĺ4  �          AQ�@�����(���p��ffC�)@�����(�>�G�@333C��                                    Bx����  �          A��@�Q���p�?�@VffC���@�Q����?�\)A\)C�                                      Bx��׀  
�          A�R@�����H@�RAhz�C��@���ȣ�@AG�A��RC���                                    Bx���&  �          A33@����Q�@G�AQ�C�5�@����R@;�A�
=C��)                                    Bx����  �          A��@�����R?�p�AK33C���@������@8��A�p�C�~�                                    Bx��r  	          Az�@s�
� ��?�Az�C���@s�
��=q@��Aw33C���                                    Bx��  T          A��@|(�� Q�?��HA-p�C�R@|(����@,��A��C�~�                                    Bx�� �  "          A{@������@*=qA�ffC�� @�����p�@eA�(�C�j=                                    Bx��/d  �          A{@�����\)@p�A`z�C��f@�����z�@J=qA�z�C�{                                    Bx��>
  �          A�@�G���z�@   AJffC�^�@�G���\@<��A��HC��f                                    Bx��L�  �          A��@��\��{?�{A"�HC�g�@��\��p�@$z�A�\)C���                                    Bx��[V  �          A�@��H�׮�Q�����C�w
@��H��G�=#�
>�\)C�\)                                    Bx��i�  �          A�H@���33����)��C��{@���33>��@AG�C��
                                    Bx��x�  �          A=q@�(���>�G�@7�C�Ǯ@�(���?��AQ�C���                                    Bx�ŇH  �          A�R@�����
=?J=q@�33C���@������?��
A9�C��{                                    Bx�ŕ�  �          A@�\)����?�  @�  C�Ff@�\)��R?���ALQ�C���                                    Bx�Ť�  "          A@����?8Q�@�{C�0�@���\?�p�A4Q�C�p�                                    Bx�ų:  �          A�@�\)��33?@  @���C�'�@�\)��{?�\A6=qC�h�                                    Bx����  T          A�\@�Q�����?�\@P��C�T{@�Q����?��
A{C���                                    Bx��І  �          A
=@����{?(��@��C�!H@������?�A,z�C�b�                                    Bx���,  �          A��@����녿(��|(�C��H@����=q>�(�@3�
C��q                                    Bx����  �          A�@�=q��녿n{��  C�@�=q���
=�?L��C��                                    Bx���x  �          A�@�
=���׾��G�C�B�@�
=����?\)@h��C�C�                                    Bx��  �          A��@�ff��  ����,(�C�8R@�ff���?!G�@�33C�=q                                    Bx���  �          AG�@"�\�Q�?Tz�@��C���@"�\�G�?�Q�AN=qC��
                                    Bx��(j  "          A?�z��(�?�p�A{C�U�?�z���@'�A��C���                                    Bx��7  �          A�@:�H���>8Q�?��C���@:�H�\)?�ffA  C���                                    Bx��E�  �          Az�@L(���\<#�
=�Q�C���@L(��G�?���@�z�C���                                    Bx��T\  
          A(�@l(���33�@  ���RC���@l(���(�>���@
�HC���                                    Bx��c  l          A�@}p���Q�����  C��@}p����R?k�@�G�C���                                    Bx��q�  T          A�
@tz����\>aG�?�C�  @tz���\)?��A��C�'�                                    Bx�ƀN  �          A
=@w�����=�Q�?(�C�=q@w���?�@��C�^�                                    Bx�Ǝ�  
�          A
=q@��H��R�L�;��
C�R@��H��z�?}p�@��HC�4{                                    Bx�Ɲ�  �          A
{@�\)��{>���?��RC��@�\)��\?�ffA
=C���                                    Bx�Ƭ@  �          A	�@�  ���=�?Tz�C�H@�  ��\)?��@߮C�/\                                    Bx�ƺ�  T          A	�@�\)��=q<#�
=�\)C�� @�\)��  ?xQ�@�\)C���                                    Bx��Ɍ  "          A	�@�{��p�>�\)?�C���@�{��=q?�(�AC��3                                    Bx���2  �          A	��@�\)��\�u�˅C��
@�\)��G�?L��@��
C���                                    Bx����  �          A	�@XQ����\�#�
��Q�C���@XQ����H>��H@P  C��=                                    Bx���~  T          A
=@   �(��O\)��z�C�h�@   ���>�p�@�RC�aH                                    Bx��$  T          A
�R@1���H���p  C�ff@1���R?(�@���C�g�                                    Bx���  T          A
=q@�G����������C�@�G���=q�#�
��z�C���                                    Bx��!p  �          A\)@��
��33�u���C�8R@��
���>��?�  C�!H                                    Bx��0  �          A
=@G�� �׿}p�����C��R@G����>B�\?��RC��f                                    Bx��>�  �          A33@e���
�J=q��
=C�B�@e����>�Q�@��C�8R                                    Bx��Mb  �          A33@l(����\�k���=qC��q@l(���(�>k�?��
C���                                    Bx��\  
�          A
�\@�ff��zᾮ{�  C��f@�ff��?@  @�
=C��3                                    Bx��j�  �          AQ�@^�R�������
=qC��H@^�R� Q�\)�fffC��q                                    Bx��yT  
�          A�@e��    �#�
C�+�@e���H?�@��RC�H�                                    Bx�Ǉ�  
�          A�@�����Q�>�{@G�C��
@�����(�?�A��C��{                                    Bx�ǖ�  �          A�
@�(���׾�����(�C��@�(���\)?Q�@��C�+�                                    Bx�ǥF  �          Az�@Fff� �ÿ��H�C��f@Fff�
=��  ����C�`                                     Bx�ǳ�  
(          A(�@33��\���f�RC�޸@33�ff�p����z�C���                                    Bx��  �          A
{@_\)��
=�����C��H@_\)��33��z���HC�Q�                                    Bx���8  T          A
=q@�����
�z��w
=C���@�����
?   @VffC��\                                    Bx����  
          A	@qG���ff�Q��d��C�p�@qG����z�H��G�C��                                    Bx���  l          A	��@\)������33C��
@\)��p���p����C�#�                                    Bx���*  �          A	p�@�p���=q��{�-G�C��{@�p���\)���Mp�C���                                    Bx���  �          A
{@L����p��2�\���\C��@L����Q�У��/�
C�8R                                    Bx��v  �          A��@g
=��\)���PQ�C��H@g
=��{�@  ��=qC��{                                    Bx��)  "          A	G�@\(���=q���R�W\)C�7
@\(���G��O\)���C��=                                    Bx��7�  �          A	p�@�\)��G����\��Q�C�h�@�\)���
=#�
>�=qC�B�                                    Bx��Fh  �          A	�@���{?O\)@��RC��H@���\)?�=qAG
=C�P�                                    Bx��U  �          A	��@��\���
���\�ۅC�aH@��\��{=���?#�
C�@                                     Bx��c�  �          A��@xQ���ff�����,Q�C��f@xQ���33��
=�5�C���                                    Bx��rZ  �          A	p�@�{��p�����陚C�@�{��      <#�
C��R                                    Bx�ȁ   �          A	G�@u�����{��\C��@u����.{��33C�^�                                    Bx�ȏ�  T          AQ�@S�
����Q���G�C��R@S�
����>��@0  C���                                    Bx�ȞL  �          A
�H?�33��z������	��C��
?�33��\�p  ��\)C��                                    Bx�Ȭ�  �          A	p�@5���ÿ�{�J�RC�
=@5��\)�(���z�C�Ф                                    Bx�Ȼ�  �          A��@S33��  ?��A(�C���@S33��{@"�\A���C��                                    Bx���>  �          A	p�@h����33@{A�=qC�+�@h���ۅ@g
=A�33C��
                                    Bx����            A33@����
=@^�RA��C���@����33@�{A�C���                                    Bx���  T          A�@�(��%�@�33BffC���@�(���\@�ffB�C���                                    Bx���0  �          A�H@�  ��Q�@�  Bz�C��=@�  �W�@��B0C��                                    Bx���  �          A�R@����{@�ffB�HC���@�����
@�
=B#�\C�
                                    Bx��|  
�          A��@��R��ff?�33A
=C��@��R��z�@\)A�Q�C��\                                    Bx��""  "          A	G�@�\)��G�>aG�?�p�C���@�\)���?��AC��                                    Bx��0�  "          A��@�\)��=q�����,(�C��f@�\)��G�?@  @���C��3                                    Bx��?n  
�          A	�@�(��ۅ��\�c�
C�H@�(���33?��@�p�C��                                    Bx��N  
�          AQ�@C33�ڏ\@~�RA�  C�
=@C33���H@���B��C�L�                                    Bx��\�  "          A\)@]p���=q@\)A�(�C�޸@]p����\@�Q�B��C�Q�                                    Bx��k`  
�          A33@ ���љ�@��BffC���@ ����p�@��
B1�C���                                    Bx��z  T          AQ�@�R���@^{A�G�C�\)@�R�׮@�z�B�\C�"�                                    Bx�Ɉ�  �          AQ�@0����=q@FffA��HC�T{@0���ָR@�Q�A�33C�,�                                    Bx�ɗR            A��@�(���G�?���A-G�C�� @�(���@1�A�(�C�b�                                    Bx�ɥ�  �          A��@q����H<��
>#�
C�AH@q���\)?��\A
=qC�k�                                    Bx�ɴ�  
�          A��z�H���@O\)A��
C�XR�z�H���@��RB ��C��                                    Bx���D  l          A	��@(��33?z�@w�C�J=@(����?�
=AP  C�~�                                    Bx����  T          A
=q@#33�
=�u���C��f@#33�p�?�ffA�C��                                     Bx����  T          A
=@6ff��R�.{����C��)@6ff��R?0��@�G�C��)                                    Bx���6  T          A
=q@Tz���G����B�\C��@Tz���?��\A
=qC���                                    Bx����  �          A
�H@7
=��33��\�[\)C�H@7
=�G��333��z�C���                                    Bx���  T          A	�@<(���\)�r�\�ԏ\C�g�@<(�����#�
����C���                                    Bx��(  �          A
ff@C33���>��
@	��C���@C33����?�\)A2�\C�޸                                    Bx��)�  �          A
�\@5�G�?J=q@���C���@5���\@��Af�RC���                                    Bx��8t  �          A�@\)�Q쾽p���C�c�@\)��?��@�(�C�q�                                    Bx��G  T          A  @<(��p��8Q���G�C���@<(����?(��@��
C��q                                    Bx��U�  �          A�R@���� ������C�l�@���׿�=q��  C�+�                                    Bx��df  T          A�?���녿p�����
C�  ?���>�Q�@'�C�{                                    Bx��s  �          A	G�@33��ff���R��C�t{@33��(��^{��G�C�K�                                    Bx�ʁ�  �          A�
@
=�
=������C���@
=�i����Q��lffC�o\                                    Bx�ʐX            A�
?�33�{��ff�=C�P�?�33�c33��ff�y  C�|)                                    Bx�ʞ�  �          A�H?�z���
��z�Q�C��=?�z��hQ����
�vG�C�@                                     Bx�ʭ�  T          A
=?�33�HQ���z��
C�
?�33�����߮�c�
C�T{                                    Bx�ʼJ  �          A�ͽ�Q���R��RB�C����Q��vff���
�|�
C�O\                                    Bx����  
�          A	p�@E������R�633C�� @E�������z�C�q                                    Bx��ٖ  �          A�@����녿��H�^�\C��\@����녿aG���{C�`                                     Bx���<  �          AQ�@�(���\)��(��	��C��q@�(���33���\(�C�z�                                    Bx����  T          A�R@�=q��{�����G�C���@�=q�ȣ׿�=q���C�˅                                    Bx���  �          A=q@�p���p��h���θRC���@�p���
=>���@	��C�q�                                    Bx��.  �          @�
=@����ʏ\���mC��q@������H�c�
��
=C�w
                                    Bx��"�  "          @�ff@�  ��=q��33��
=C�H�@�  ��
=�R�\�îC�J=                                    Bx��1z  �          A�@����B�\������(�C�H�@����mp��]p���z�C��
                                    Bx��@   T          Az�@�
=���U��Q�C�:�@�
=�5��8����G�C�J=                                    Bx��N�  �          A33@�33��z��   ���\C��
@�33�z��
�H�vffC�^�                                    Bx��]l  <          A��A=q������(��>�RC��\A=q�������5��C���                                    Bx��l  
�          Az�A
=��ff�����
C�|)A
=��G������Q�C��                                    Bx��z�  �          A33A
{�#�
��\)�;�C��)A
{�6ff����
�HC��)                                    Bx�ˉ^  �          A�
A�
��{��p�����C���A�
���k����C�Q�                                    Bx�˘  T          A�HA녾�׿\)�^�RC��=A녿\)���9��C�AH                                    Bx�˦�  �          AffA(���(����R��Q�C�aHA(���  �#�
��=qC�G�                                    Bx�˵P  
          Az�A{�&ff��(��(��C��
A{�(Q�=#�
>�=qC��q                                    Bx����  :          A��A\)�H�ÿ(��p��C�{A\)�L�ͽ#�
�k�C���                                    Bx��Ҝ  �          A\)A���%�>\)?fffC��HA��� ��?��@p  C�{                                    Bx���B  T          A
=A�ÿ��?@  @�ffC��A�ÿ��?xQ�@�=qC�k�                                    Bx����  �          A{A
=��p�?�ff@�33C�NA
=���\?�ffA�RC��R                                    Bx����  �          A(�A���  ?��\@�G�C�U�A����?���A\)C�H                                    Bx��4  
�          A�A33��p�?�ffA�C�aHA33��(�?˅A#33C�7
                                    Bx���  �          A
=A
�\����?��@�\)C���A
�\��(�?�A=qC�`                                     Bx��*�  �          AffAz��(�?��
A
=C�}qAz��z�?�{AAC�|)                                    Bx��9&  "          AG�A
=�k�@G�Ao
=C��A
=�   @��A|��C�Q�                                    Bx��G�  T          A��Az���
?�z�A-G�C��AzῚ�H?�z�AG�
C��R                                    Bx��Vr  T          AQ�A ���!G�@Ax  C�O\A ����
@0��A���C��R                                    Bx��e  �          A\)@�
=��(�@6ffA���C���@�
=�b�\@c33A�\)C��=                                    Bx��s�  �          A	�@�=q����=�G�?B�\C�8R@�=q����?�ffA�C��H                                    Bx�̂d  �          A��@y����\)�(�����
C�,�@y����\)?+�@�(�C�,�                                    Bx�̑
  �          A��@�{��
=�=p����HC�=q@�{�߮?��@r�\C�5�                                    Bx�̟�  �          A�@U�������33C�
@U���  >L��?�
=C��
                                    Bx�̮V  �          A
=@��H�Ǯ@p�A�33C�.@��H��@dz�A�{C�y�                                    Bx�̼�  
�          A{@�=q��@z�A�z�C���@�=q���@Tz�A��HC���                                    Bx��ˢ  �          Aff@��\��33?��A��C�aH@��\����@A��
C�                                    Bx���H  �          A�@�ff���?(�@��C�o\@�ff��G�?��AT��C��)                                    Bx����  �          A33@h����\�L�;�p�C�0�@h����
=?�ffA{C�^�                                    Bx����  �          A�@x����{���Ϳ.{C�7
@x����\?�p�A
�\C�e                                    Bx��:  �          A��@�{��  ����=qC�p�@�{��녿����C��                                    Bx���  �          A��@�G����?��HA/�C��H@�G���  @�\A�
=C���                                    Bx��#�  �          A(�?޸R��(�@�  B:�HC���?޸R��p�@�z�Bb��C�P�                                    Bx��2,            A��?�z���p�@�(�B  C��)?�z���{@�G�B/�
C�C�                                    Bx��@�  "          A��?��H��@g�A�G�C��?��H���@�B��C�=q                                    Bx��Ox  T          A���=q��p�@8��A��HC�����=q��  @�Q�A��\C��                                     Bx��^  	H          A\)>aG�� z�?�
=A!�C�Ǯ>aG����
@9��A�ffC��3                                    Bx��l�            A�R@�H��\@%A���C��@�H��
=@z�HA��C���                                    Bx��{j  �          A�H@C�
���?���A�HC�h�@C�
��z�@+�A�z�C��R                                    Bx�͊  �          Aff@\)���H@\)A�ffC�\)@\)��  @tz�A߮C��                                    Bx�͘�  �          Aff@@  ���?���A$  C��@@  ��  @3�
A�C���                                    Bx�ͧ\  �          A�R@U��G�?��A.{C�L�@U���
@8Q�A�G�C��{                                    Bx�Ͷ  �          A�@!G���
=?�{@�G�C�@!G���(�@"�\A�ffC�j=                                    Bx��Ĩ  �          A�H@����?�z�A�\C��q@����H@5A��\C�Ff                                    Bx���N  �          A\)@7���ff?�G�AG
=C��@7���\)@HQ�A�(�C�'�                                    Bx����  �          A�\@S�
��?�z�A<Q�C�Ff@S�
�ٙ�@?\)A��RC��R                                    Bx���  �          A�\@��\��Q�@�A�p�C��@��\��{@dz�A�z�C�k�                                    Bx���@  �          A�R@�����@&ffA�33C�˅@�����=q@qG�A܏\C�                                      Bx���  �          A  @�\)��
=@?\)A���C��q@�\)��G�@��A�  C�t{                                    Bx���  �          A��@�ff���@&ffA�{C�XR@�ff��ff@o\)A���C���                                    Bx��+2  �          A�
@�  ��(�@�HA�ffC��f@�  ���@_\)A�{C�XR                                    Bx��9�  �          A�@��H���
?���A
=C�P�@��H��G�@��Ax��C�33                                    Bx��H~  �          AQ�@�����  >�z�@�\C��@������H?�=qA��C���                                    Bx��W$  �          A��@��
���>�Q�@�RC�/\@��
��\)?\A(Q�C��)                                    Bx��e�  �          A�@�ff�����G��:�HC��
@�ff����?�G�@�{C��                                    Bx��tp  �          Ap�@��H����G��:�HC�C�@��H���H?��\@�=qC�z�                                    Bx�΃  �          AG�@��H��\)�:�H���C���@��H��ff���H�ap�C�,�                                    Bx�Α�  �          @�\)@�p������(�����HC��f@�p���{�У��Dz�C��f                                    Bx�Πb  �          A��@�z���{?��@x��C��)@�z���
=?�(�AD��C�]q                                    Bx�ί  �          A�H@��
��(�?�A(�C�p�@��
��G�@�HA�C�*=                                    Bx�ν�  �          Aff@�=q��{?333@�ffC�*=@�=q��?�(�A`  C���                                    Bx���T  �          A\)@�{�ƸR?���@�{C���@�{��(�@33A��\C���                                    Bx����  �          A�
@�������?�33AUp�C�t{@�����p�@EA�ffC��=                                    Bx���  
�          AQ�@��
����?n{@ϮC��3@��
���@Ak�C�E                                    Bx���F  �          A�@�G������z��!�C�ff@�G���녿����\)C��                                    Bx���  �          A33@���=q��33�W\)C���@���33����陚C��R                                    Bx���  �          A�\@ʏ\�����\)�7
=C�B�@ʏ\���\�&ff��33C��                                    Bx��$8  �          A�@�{���H��=q�R�\C�5�@�{��=q���p  C��                                    Bx��2�  �          Ap�@�\)�����ff�s�C��
@�\)��ff���\��\C��\                                    Bx��A�  T          A Q�@��
�������
=C��@��
�����{�G�C��                                    Bx��P*  �          @��R@������Ϳ�p��hQ�C�.@������\(���
=C��                                     Bx��^�  �          A   @�G������  ����C�'�@�G���(���
=�Q�C�G�                                    Bx��mv  �          A ��@��H����(Q�����C�� @��H��=q�˅�6{C���                                    Bx��|  �          A (�@�  ��33�^�R���
C��H@�  ����R��p�C��                                    Bx�ϊ�  �          @�{@����33@33A��C�� @����G�@XQ�A��
C���                                    Bx�ϙh  T          @�@�33��\)@333A��C�f@�33���\@y��A�\C��                                     Bx�Ϩ  �          @�
=@�=q���R@�A�  C�l�@�=q��p�@R�\A��C���                                    Bx�϶�  �          @�{@����ff@N{A���C��R@����
=@���B�
C��f                                    Bx���Z  �          @�z�@�ff���@C33A�Q�C�� @�ff���
@���A���C�~�                                    Bx���   �          @�p�@�  ��
=@#33A��C���@�  ���@j=qA�
=C�33                                    Bx���  �          @�p�@:=q��{@���BG�C�� @:=q���R@�ffB;G�C�N                                    Bx���L  �          @�Q�@^{�x��@�{B=�HC��q@^{�.{@ʏ\B\��C���                                    Bx����  �          @�
=@W����
@�ffB6C���@W��>�R@�z�BW(�C�=q                                    Bx���  �          @��@�����@�=qB��C�H�@��^�R@��B<�\C��                                    Bx��>  �          A   @e���=q@���B=qC�^�@e���(�@���B'�C�ٚ                                    Bx��+�  �          @�@;��ٙ�@/\)A��\C���@;���z�@�Q�A�=qC���                                    Bx��:�  �          @�
=@7���=q@  A��C�\@7��Ϯ@e�A�Q�C��                                    Bx��I0  �          @�@{���z�@\��Aڏ\C��@{����@�B�C�9�                                    Bx��W�  �          @�@�=q��z�@b�\A�  C��@�=q��33@�=qBz�C���                                    Bx��f|  �          @�
=@r�\��\)@s33A�\)C�Z�@r�\��(�@�G�B��C���                                    Bx��u"  �          @��R@�����@�\AuC�1�@����
=@J�HA�(�C�o\                                    Bx�Ѓ�  �          @�p�@�z��ʏ\?=p�@��HC��
@�z����?��HAo
=C�(�                                    Bx�Вn  �          @��H@�  �ƸR�xQ���C�ff@�  ����>��?�p�C�AH                                    Bx�С  �          @�
=@�{�����Y���ҏ\C�]q@�{��33>�33@+�C�AH                                    Bx�Я�  �          @�G�@�����=q<�>uC�O\@�����{?�G�A
=C��3                                    Bx�о`  �          @���@����Ǯ>�33@-p�C���@������?���A@��C�:�                                    Bx���  �          @��@��
���R@Q�A���C�W
@��
��@P  A��C��)                                    Bx��۬  �          @�  @r�\����@8Q�A���C���@r�\��(�@}p�A���C�8R                                    Bx���R  �          @�@%���Q�@^{A�G�C��R@%���
=@�=qB��C�"�                                    Bx����  �          @�R@+���{@{A�Q�C�H�@+����\@k�A���C�S3                                    Bx���  �          @�
=@B�\��G�@H��A�33C�U�@B�\���@�Q�B\)C�޸                                    Bx��D  �          @�z�@[�����@W
=Aڣ�C��=@[���Q�@�(�B�
C��H                                    Bx��$�  �          @��@N�R��  @l��A�33C�0�@N�R��p�@��RB��C�O\                                    Bx��3�  �          @�(�@$z�����@��B�
C�p�@$z����
@��B0{C���                                    Bx��B6  �          @�@W���{@o\)A��C��@W���33@��B�RC�)                                    Bx��P�  �          @�ff@R�\��@^�RA߮C��@R�\��z�@���B�C��{                                    Bx��_�  �          @�  @|(����H@a�A�{C�7
@|(���G�@�  BQ�C�u�                                    Bx��n(  �          @�{@�Q����
@dz�A��
C�
=@�Q����\@�  BG�C�l�                                    Bx��|�  �          @�{@������\@}p�B �C���@����mp�@���B =qC��{                                    Bx�ыt  �          @�ff@�=q����@�G�BQ�C�q�@�=q�Z=q@���B \)C���                                    Bx�њ  �          @���@�(����H@p  A��RC��{@�(��qG�@�=qB�C��f                                    Bx�Ѩ�  �          @��@tz����@XQ�A��C��@tz���\)@��HBp�C�9�                                    Bx�ѷf  T          @陚@�33���@<��A�G�C�  @�33��=q@z�HBG�C��3                                    Bx���  �          @��
@dz�����@�\)B*�C���@dz��?\)@�BJ�C��                                    Bx��Բ  �          @���@g����H@|(�B �
C���@g���
=@��B$p�C�N                                    Bx���X  �          @�\)@)�����
@ ��A���C���@)������@j=qA���C��
                                    Bx����  �          @�
=@8�����@<��A�33C��@8����
=@���B��C�~�                                    Bx�� �  �          @�\)@QG����
@J=qA�\)C�R@QG���z�@�ffB  C�޸                                    Bx��J  �          @�@�Q����\@9��A�
=C��q@�Q���p�@uB �C��q                                    Bx���  �          @�@X����ff@Tz�A�
=C��{@X����{@��\B  C��                                    Bx��,�  �          @�\)@Mp����\@l��A�C��@Mp���  @�p�B ��C���                                    Bx��;<  �          @�{@Z=q���@P  Aי�C��@Z=q��p�@�  B33C�                                    Bx��I�  �          @�@j=q��\)@<��A�G�C��q@j=q����@~{B\)C��=                                    Bx��X�  �          @�@e���33@2�\A�ffC�K�@e���{@u�B �C��3                                    Bx��g.  T          @�@'���@AG�A�  C���@'���
=@��
B��C�Q�                                    Bx��u�  �          @��@g����H@C33A���C��@g���z�@�G�B	�HC���                                    Bx�҄z  �          @�33@mp����@>{A�ffC��=@mp����@{�B�C��{                                    Bx�ғ   �          @�33@O\)��(�@G
=A���C���@O\)��p�@�33BG�C�b�                                    Bx�ҡ�  �          @�@(���(�@N�RA�  C��R@(���z�@�Q�B�
C�B�                                    Bx�Ұl  �          @�
=@�p����H@�A�(�C���@�p����H@A�A�z�C�!H                                    Bx�ҿ  �          @�=q@�{��33@ffA��C��=@�{��33@<��A�p�C�*=                                    Bx��͸  �          @���@�p���Q�@
=A�(�C��q@�p�����@<��A�33C�e                                    Bx���^  �          @�\)@�����@�\A�G�C��@����y��@G
=Aә�C��q                                    Bx���  �          @�Q�@���tz�@�
A�z�C�B�@���S33@@��A�=qC�33                                    Bx����  �          @��
@�G���Q�@�A�Q�C�q�@�G���
=@Mp�A���C��                                    Bx��P  �          @�33@\(���=q@p�A�  C�t{@\(����@\��A�(�C�                                      Bx���  �          @ٙ�@AG���z�@)��A�C��)@AG�����@i��B�C�#�                                    Bx��%�  �          @�{@���p�>L��?��HC�@�����?��A33C�(�                                    Bx��4B  �          @�(�@�\)���?�ffAT��C�#�@�\)���H@\)A�C�P�                                    Bx��B�  �          @�33@�����  >\)?��HC�w
@�����(�?�{A=qC���                                    Bx��Q�  �          @ۅ@�=q���׿G���G�C�"�@�=q���H=�G�?uC��\                                    Bx��`4  �          @ڏ\@�33���R?�ffAR�\C�)@�33���@!�A��RC�:�                                    Bx��n�  �          @ڏ\@~{��ff?�Q�A�  C��3@~{��
=@:=qA��C�
=                                    Bx��}�  �          @�33@�
=��=q?#�
@���C���@�
=��33?�{A[�C�W
                                    Bx�ӌ&  �          @�(�@�{��
=�����G�C�33@�{����?\(�@�C�c�                                    Bx�Ӛ�  �          @�  @�{���
�\)���\C���@�{��(�>��H@���C��                                    Bx�өr  �          @�z�@�(���  ��
=�a�C�j=@�(����?\)@�C�s3                                    Bx�Ӹ  �          @�  @�33��z��-p�����C��)@�33��=q��G��qG�C�w
                                    Bx��ƾ  �          @�p�@�Q���=q�\)��33C�޸@�Q������G��/
=C���                                    Bx���d  �          @�
=@}p���G��G���(�C�\@}p����H��G��
=qC�Ff                                    Bx���
  �          @���@u���
�����C�q@u��������@  C�q                                    Bx���  �          @׮@�  ��\)�����C�]q@�  ��G���=q�z�C���                                    Bx��V  �          @ڏ\@�����G��^�R���HC�xR@������
=�\)?z�C�<)                                    Bx���  �          @�  @�(���ff�Q���=qC�9�@�(���Q�>L��?�C�\                                    Bx���  �          @θR@{��33?^�R@���C�p�@{��=q?�p�A��C���                                    Bx��-H  �          @θR@Vff��\)>���@>�RC��3@Vff���?���AF�HC��                                    Bx��;�  �          @��H@w����ýu���C�#�@w���{?s33A{C�^�                                    Bx��J�  �          @���@U��
=>��
@0  C�&f@U����?�z�AD��C��f                                    Bx��Y:  �          @�Q�@��\����{�7
=C���@��\��z�?(��@�z�C��                                    Bx��g�  �          @ڏ\@��
��ff��R��Q�C���@��
��\)>�{@6ffC�s3                                    Bx��v�  �          @ָR@�����׾�(��n�RC�(�@����Q�>��H@�ffC�,�                                    Bx�ԅ,  �          @љ�@�p���=q��ff�|(�C��
@�p���=q>�(�@s33C���                                    Bx�ԓ�  �          @ҏ\@��\���\�xQ����C�\@��\��{������C���                                    Bx�Ԣx  �          @�\)@������
?�
=A�C���@����n{@�A��\C��                                    Bx�Ա  �          @Ӆ@x�����
@X��A�p�C��=@x���X��@��
BffC�n                                    Bx�Կ�  �          @љ�@������
@EA�\)C�1�@����\��@tz�B�HC���                                    Bx���j  �          @Ϯ@k����@>{AڸRC��3@k��p��@p��B
=C�7
                                    Bx���  �          @ʏ\@mp����
@*�HA�(�C�(�@mp��q�@]p�B��C�:�                                    Bx���  �          @Ǯ@Q�����@.�RAϮC��=@Q��|(�@c33B
��C���                                    Bx���\  �          @�ff@�{�5�>�33@XQ�C���@�{�-p�?aG�A��C�8R                                    Bx��	  �          @���@����
=��\��ffC��@����=q�L�Ϳ   C��=                                    Bx���  �          @ȣ�@��
�J�H�������C���@��
�N�R<#�
=���C��H                                    Bx��&N  �          @Ǯ@�z��o\)>��
@;�C�J=@�z��g�?��\AQ�C��                                    Bx��4�  �          @�Q�@�33�s�
>���@n�RC��@�33�j�H?�\)A&�\C�u�                                    Bx��C�  �          @��@��R�u>�G�@\)C�"�@��R�k�?�z�A)��C���                                    Bx��R@  �          @���@�
=���H���
�L��C���@�
=����?@  @�p�C��=                                    Bx��`�  �          @�@���\)>��@��HC���@���u�?��HA5��C�O\                                    Bx��o�  �          @ə�@�z��A�?�A�
=C���@�z��(Q�@Q�A��C�q�                                    Bx��~2  �          @�33@���<(�@W
=B 33C�^�@�����@w
=B  C���                                    Bx�Ռ�  �          @�33@�
=�I��@aG�B�HC�� @�
=�(�@���B�HC�                                    Bx�՛~  �          @ʏ\@����@  @O\)A�z�C�:�@����@p  BG�C�XR                                    Bx�ժ$  �          @���@mp��   @�Q�B.�C��@mp���33@��BC��C��q                                    Bx�ո�  �          @Ǯ@����-p�@q�B\)C�k�@��Ϳ���@�
=B*�\C�e                                    Bx���p  �          @�  @�33�9��@mp�B�C�b�@�33�	��@�{B(\)C�*=                                    Bx���  �          @�ff@����=p�@5�Aܣ�C�o\@����Q�@U�B=qC�*=                                    Bx���  �          @�Q�@���z�>\@l(�C���@���p�?O\)@��C��                                    Bx���b  �          @ƸR@�  �1G��{���C���@�  �HQ��z��{�
C���                                    Bx��  �          @ȣ�@�
=�8Q���\���C��\@�
=�P�׿��H���\C�]q                                    Bx���  
�          @���@�Q��h���������C���@�Q����׿��~�\C�S3                                    Bx��T  �          @�Q�@�����33�Ǯ�hz�C��
@�����녿8Q���C�'�                                    Bx��-�  �          @Ǯ@�p��\(��  ���C��@�p��s33��=q�j�RC�n                                    Bx��<�  �          @�ff@�z��#�
������C�Ǯ@�z��;���  ����C�&f                                    Bx��KF  �          @Å@����7
=�k��
�HC���@����>�R��p��`��C�K�                                    Bx��Y�  �          @�G�@�p��2�\=�Q�?^�RC��@�p��.�R?��@���C�{                                    Bx��h�  �          @���@��\���@EBp�C��@��\���@UB�C��)                                    Bx��w8  �          @�ff@7��@  @��
Bo�HC��3@7�>W
=@�p�Bs��@�
=                                    Bx�օ�  �          @��H@`  ��?��A�\)C��@`  ��p�@ffA�C��=                                    Bx�֔�  �          @���@���33?�\)AUG�C�L�@��� ��?�\A��\C��f                                    Bx�֣*  �          @�{@�{�7��p����C���@�{�?\)�����}p�C�e                                    Bx�ֱ�  T          @��H@��
�Tz�?�{A}�C�!H@��
�>{@(�A�=qC��                                     Bx���v  T          @�@�z��H�ÿ8Q��ٙ�C�P�@�z��Mp�����  C�                                      Bx���  �          @�33@~{�8Q��]p��C�@~{�]p��8Q�����C�z�                                    Bx����  �          @���@���  ���\�?33C���@���(��W
=��33C�%                                    Bx���h  �          @ȣ�@�\)��G��	������C���@�\)�У׿����ffC�                                    Bx���  �          @���@�����;���Q�C�}q@����  �2�\����C��                                    Bx��	�  �          @ƸR@�Q�>�33�5��ᙚ@s�
@�Q�.{�5��
=C�{                                    Bx��Z  �          @�G�@g��]p��p����RC�'�@g����\�Dz����C�Ǯ                                    Bx��'   �          @�=q@'�����������C�g�@'���\)�P  ���C�}q                                    Bx��5�  �          @�ff@������p  �G�C�@���(��7����
C��f                                    Bx��DL  �          @�G�>.{��{�w
=��C���>.{����:=q����C���                                    Bx��R�  T          @�=q?���=q��Q��+=qC�B�?���Q��W����C���                                    Bx��a�  �          @���?��������Q��5��C��=?�����\)�j�H���C�p�                                    Bx��p>  
�          @�(�?333��{���2Q�C�s3?333��p��c�
�	ffC���                                    Bx��~�  �          @��H���������\�4�RC�uÿ���\)�`  �C���                                    Bx�׍�  �          @�=q?��H������Q�� �
C��3?��H����H������C���                                    Bx�ל0  �          @�33�����Q���(��0ffC�;����
=�`  �Q�C�.                                    Bx�ת�  �          @˅?�ff�����(��#�C��f?�ff�����O\)����C��3                                    Bx�׹|  �          @�(�?&ff��p��{���C���?&ff��G��@  ��{C�T{                                    Bx���"  �          @�{?z���33��\)�/��C���?z���=q�e�\)C��                                    Bx����  �          @�p��
=�c33����_\)C�Ff�
=�����H�7
=C�4{                                    Bx���n  �          @�G������{����(�\C�s3�����z��]p�� ��C�E                                    Bx���  �          @�  �����  �����$��C�G�������XQ�����C�R                                    Bx���  �          @θR��\�����Q��%�C����\��\)�Vff����C�Y�                                    Bx��`  �          @θR<��
��=q�����3z�C��<��
��G��j�H�
=C��                                    Bx��   
�          @�ff�\��ff�����)ffC���\��z��Z=q�(�C���                                    Bx��.�  �          @�{��R��z������,��C�*=��R���H�a��  C��=                                    Bx��=R  �          @�{�z���z���p��9�C�5ÿz���z��u��
=C��H                                    Bx��K�  �          @��Ϳ
=q������{�JG�C�4{�
=q���\����"p�C��{                                    Bx��Z�  �          @�Q�fff�z=q�����Hz�C���fff����G��!Q�C���                                    Bx��iD  �          @�Q��R��  ����5�HC�\��R��\)�p���G�C���                                    Bx��w�  �          @�ff�z�H��=q���
�E=qC�AH�z�H������H�\)C�O\                                    Bx�؆�  �          @Ϯ�#�
�s�
��\)�W��C��\�#�
�����  �/�HC�f                                    Bx�ؕ6  �          @�  ��{�Y����ff�b��C�&f��{�����G��;p�C��
                                    Bx�أ�  �          @��H��  �aG������Rp�C~0���  ��G�����,{C�u�                                    Bx�ز�  �          @�
=��=q�g
=����K  C}aH��=q���H�z�H�$�C��                                    Bx���(  �          @�  �s33�p  ���R�=�
C�q�s33����`����C���                                    Bx����  �          @�{�Y���W���{�O��C�׿Y�����\�s33�)z�C�!H                                    Bx���t  
�          @�녾.{�r�\��=q�O=qC���.{��������({C��                                    Bx���  �          @�=q�\)��
=�%�ʏ\C�>��\)���\��\)�x��C�J=                                    Bx����  �          @�33�=q�dz���(��<Q�Ck�q�=q�����}p��
=Cp                                    Bx��
f  �          @�>���hQ��z�H�<�\C�1�>������P  ��C�f                                    Bx��  �          @���@�33���?��A�  C��\@�33��\)?��HA�{C��R                                    Bx��'�  �          @�G�@��H����@��A��RC�c�@��H=#�
@
=qA���>�33                                    Bx��6X  �          @�(�@��E@0��A�\)C�H�@��%�@N�RA�p�C���                                    Bx��D�  �          @�p�@�\)�E@33A�p�C��{@�\)�,��@"�\A���C�S3                                    Bx��S�  �          @���@�z���?�G�AV=qC��@�z���?��A�Q�C�C�                                    Bx��bJ  �          @ҏ\@��ÿ��
?0��@��C��@��ÿ��?z�HA	G�C��3                                    Bx��p�  �          @���@��
�	��?�@�Q�C�T{@��
�G�?k�A��C�ٚ                                    Bx���  �          @�\)@�=q� �׿:�H�ӅC�W
@�=q�&ff���R�1�C���                                    Bx�َ<  �          @�{@�(��b�\�����{C��@�(��u�\�\z�C���                                    Bx�ٜ�  
�          @�p�@����������=C��@�����
�s33�Q�C�N                                    Bx�٫�  �          @���@�
=�G�����Q�C���@�
=�
=q�5����C��                                    Bx�ٺ.  �          @Ϯ@��ÿ�{�0������C�޸@��ÿ��H�����\C�u�                                    Bx����  �          @�p�@�(���>�33@@��C���@�(���
=>�
=@i��C�33                                    Bx���z  �          @��@�
=��33�����G�C���@�
=��(����
�5C�@                                     Bx���   �          @���@����33��z��D  C���@����  �����RC��3                                    Bx����  �          @�(�@��Ϳ�
=�\�T  C�N@����	����Q��%�C�aH                                    Bx��l  �          @�@ʏ\�޸R��
=�#33C�Q�@ʏ\��z�c�
��(�C���                                    Bx��  �          @Ӆ@��
��Q쿁G����C���@��
��=q�E�����C��                                    Bx�� �  �          @љ�@�p���z����G�C��f@�p����H��=q�
=C��                                    Bx��/^  �          @Ӆ@�=q��R>�  @�C�J=@�=q�z�>�33@AG�C��                                     Bx��>  �          @�\)@�33���?���A ��C�޸@�33��z�?�Q�A)��C��{                                    Bx��L�  �          @ҏ\@�
=�L��?�A#�C��=@�
=>\)?�z�A"�\?�p�                                    Bx��[P  �          @ҏ\@�  ��\?333@ÅC���@�  �Ǯ?E�@׮C�G�                                    Bx��i�  �          @У�@�{�0��>�33@G�C��@�{�!G�>�@���C�7
                                    Bx��x�  �          @�
=@��;�?G�@�
=C���@��;���?Y��@���C��f                                    Bx�ڇB  �          @�
=@�G�    ?���AF�R<#�
@�G�>aG�?�\)AD(�?��R                                    Bx�ڕ�  �          @�G�@ȣ�?(��?�\)Af=q@���@ȣ�?h��?�  ATQ�A�                                    Bx�ڤ�  �          @У�@˅�@  ?�\)A\)C�� @˅��?�p�A.=qC�s3                                    Bx�ڳ4  �          @Ϯ@��ÿ�ff?uA��C�#�@��ÿ��?�33A$(�C�޸                                    Bx����  �          @�\)@˅��?L��@��
C�n@˅��G�?aG�@��\C��                                    Bx��Ѐ  �          @θR@ə��aG�?��A(�C�@ə��5?�z�A&{C�Ǯ                                    Bx���&  �          @�
=@�  ��p�?�G�Az�C�l�@�  ���?�Q�A*{C�.                                    Bx����  �          @�33@��ÿ��?\(�@�C��@��ÿ��?���A{C��\                                    Bx���r  T          @��@�  <��
?�\)AiG�>#�
@�  >�=q?���Af{@�                                    Bx��  �          @��H@��?\(�@
�HA�@�p�@��?�
=@ ��A�Q�A-�                                    Bx���  �          @���@ə�?���?���A~�\A�\@ə�?��?��AdQ�A?�                                    Bx��(d  �          @��@�
=?��@�RA�{A�H@�
=?�z�@�\A�  ATQ�                                    Bx��7
  T          @�  @�=q?z�H@A�  A�H@�=q?��
?�z�A�G�A?33                                    Bx��E�  �          @У�@��?0��?�(�AP��@�
=@��?fff?���A?�A{                                    Bx��TV  �          @��
@У�>��H?�  A
�H@�=q@У�?#�
?h��@�{@�=q                                    Bx��b�  
�          @��@�z�?(�?�ffA7�@��@�z�?L��?���A(��@���                                    Bx��q�  
(          @���@�{>�33?Q�@�  @I��@�{>��?B�\@�ff@�{                                    Bx�ۀH  T          @�ff@�p��L��?\)@��RC��=@�p�=#�
?\)@�
=>���                                    Bx�ێ�  �          @�z�@��=�Q�?h��A�H?Tz�@��>aG�?aG�@�
=@�                                    Bx�۝�  �          @�ff@�(�?�?0��@�(�@���@�(�?(�?��@�z�@�                                      Bx�۬:  �          @�ff@�\)��?�33Am�C���@�\)>W
=?��Ak33?�Q�                                    Bx�ۺ�  �          @���@ə��#�
?z�HA�C��@ə�>�?xQ�A�
?�
=                                    Bx��Ɇ  �          @�ff@��þ��?��A<z�C��)@��þ�=q?�\)ADQ�C��                                     Bx���,  �          @�p�@��Ϳ#�
?�z�Ap��C��@��;Ǯ?޸RA|(�C�.                                    Bx����  �          @�{@�p�<#�
?�G�A33=u@�p�>��?޸RA|z�@�H                                    Bx���x  
�          @�@�{?
=q?�\)Aip�@�ff@�{?B�\?\A[�@��                                    Bx��  T          @���@�(�>���?��
A�33@E�@�(�?z�?��HAx��@��                                    Bx���  �          @�(�@��>�{?�=qAep�@HQ�@��?\)?�G�A[�
@�                                    Bx��!j  T          @ʏ\@�Q�#�
?�Q�A���C�9�@�Q�=�G�?�Q�A�33?�=q                                    Bx��0  T          @ə�@�녾�{?�z�At��C�c�@�녽���?�Q�Ay��C���                                    Bx��>�  T          @�p�@ƸR�(�?�
=AO
=C�0�@ƸR����?�G�AY��C�#�                                    Bx��M\  �          @��@�녿+�?Y��@���C���@�녿
=q?p��A�RC���                                    Bx��\  �          @�(�@�ff�u?�\)A!�C���@�ff�J=q?��RA3�
C�Y�                                    Bx��j�  T          @�=q@�녿��?�G�A�C��=@�녿��R?�Q�A.�\C�<)                                    Bx��yN  �          @�=q@�  ��G�?��A;�
C��@�  ����?�p�AX��C��\                                    Bx�܇�  �          @ə�@�G���33?(�@��C�Z�@�G���ff?Tz�@��HC�˅                                    Bx�ܖ�  �          @ȣ�@�  ��\)?=p�@���C�t{@�  ��  ?s33AQ�C���                                    Bx�ܥ@  �          @���@��
�ff>���@Dz�C�*=@��
��?�R@�C�q�                                    Bx�ܳ�  �          @�33@��H�Y�������hQ�C��H@��H�Z�H=u?z�C���                                    Bx��  �          @��@�33�i����\��(�C�0�@�33�l(��#�
��Q�C�\                                    Bx���2  �          @ʏ\@��\�k������  C��@��\�n�R���Ϳp��C�޸                                    Bx����  �          @��@���QG�������C���@���P��>��
@6ffC��f                                    Bx���~  �          @��
@���J�H���
�@  C��)@���J=q>�{@E�C�                                    Bx���$  �          @��H@�33�5�z����C���@�33�8�þL�Ϳ�=qC�Z�                                    Bx���  �          @�Q�@�\)�<(�>�?�Q�C��@�\)�9��?�@���C��                                    Bx��p  �          @���@��
�2�\=u?�C�Ǯ@��
�0��>�
=@w�C���                                    Bx��)  
�          @��@�z��33�Q���ffC�h�@�z���ÿ
=q��C�f                                    Bx��7�  T          @���@�����׿k���C�T{@����
=��R��C��f                                    Bx��Fb  �          @ȣ�@�  ���<#�
=�\)C�q�@�  ��>���@?\)C���                                    Bx��U  �          @Ǯ@�
=�����
=�vffC��)@�
=�����Ϳs33C�w
                                    Bx��c�  �          @�G�@�=q���@  ���HC�ff@�=q��G��\)��33C�f                                    Bx��rT  �          @�=q@�\)�fff�����j�HC�޸@�\)�s33��z��%C���                                    Bx�݀�  �          @��H@ȣ׿+��#�
��Q�C��
@ȣ׿=p������{C��                                     Bx�ݏ�  �          @���@�\)��p����H�X��C�z�@�\)��ÿ����1�C��)                                    Bx�ݞF  �          @ə�@�=q���\��G��9p�C�7
@�=q����\)�$��C��f                                    Bx�ݬ�  �          @�Q�@�=q���ÿ���B=qC�N@�=q��(�������\C���                                    Bx�ݻ�  �          @Ǯ@�  ���H��(��4  C�G�@�  ��{����  C��H                                    Bx���8  �          @��@��R��녿��H�V�HC�q�@��R�Ǯ���
�;�C���                                    Bx����  �          @�ff@�����R�����k\)C��@����=q���
�?�C�Ff                                    Bx���  �          @�z�@�33�,(���=q�Hz�C�� @�33�5�}p����C�f                                    Bx���*  
�          @��
@���9�����
�g�
C�XR@���C�
��z��/�C���                                    Bx���  �          @��@����W
=��{�r=qC�˅@����a녿�Q��2{C�!H                                    Bx��v  �          @�=q@��J=q���
�i��C���@��Tzῐ���,��C�f                                    Bx��"  �          @�Q�@�(��k��������C���@�(��w���=q�MG�C�Ff                                    Bx��0�  �          @��@�ff��  ��
���C�L�@�ff��\����(�C�f                                    Bx��?h  �          @�=q@�G����33����C�)@�G��,(����R���C���                                    Bx��N  �          @�  @�z��J=q�z���=qC��)@�z��X�ÿ�Q�����C��                                    Bx��\�  �          @�G�@����=p�� ���Ə\C�~�@����N�R�	����=qC�P�                                    Bx��kZ  �          @�=q@�\)�J=q���p�C�'�@�\)�XQ���H��
=C�>�                                    Bx��z   �          @��@����R��R��z�C���@���.�R��
=���C�|)                                    Bx�ވ�  �          @�{@�Q��!��
=���C���@�Q��2�\�33��G�C��                                    Bx�ޗL  
�          @��
@���8Q��{���C��)@���G
=�����(�C��3                                    Bx�ޥ�  T          @��@z=q�������Z�RC��)@z=q��G��h����C�*=                                    Bx�޴�  �          @�  @�{�vff�#�
���
C���@�{�x�þL�Ϳ��HC���                                    Bx���>  �          @�p�@��R�Tz�O\)�Q�C���@��R�X�þ�(���z�C�c�                                    Bx����  �          @��H@��
�
=q�У���(�C�
=@��
�������eG�C�9�                                    Bx����  �          @�33@���G���z��j�\C��)@���
=q��
=�C
=C�G�                                    Bx���0  �          @��@����  ���
�R�RC�
=@���У׿����5C�aH                                    Bx����  �          @��
@�G��aG�����_�
C�C�@�G����\��p��N{C��                                    Bx��|  �          @���@��׿����G���(�C�  @��׿��
��33���RC��
                                    Bx��"  �          @�\)@\(��p���{���
C�/\@\(��)����=q��(�C�.                                    Bx��)�  �          @�G�@%��33�����C�f@%���
=#�
>�C��R                                    Bx��8n  �          @��\@/\)��  �
=q���
C��q@/\)���ü�����C���                                    Bx��G  �          @�Q�@@  �z�H��ff��(�C��R@@  ���\��\)�r=qC�#�                                    Bx��U�  �          @��\@X���Tz�>�?˅C���@X���R�\>�@�=qC���                                    Bx��d`  �          @���@\)�9����G���G�C��R@\)�9��>L��@�C���                                    Bx��s  �          @�@������R�G���\)C��f@����(�����\)C�Ǯ                                    Bx�߁�  �          @�G�@[��=p��@���p�C��
@[��P���,(�����C�<)                                    Bx�ߐR  �          @�(�@0  ��p������x��C��=@0  �����}p��(Q�C��\                                    Bx�ߞ�  �          @��@'���������\)C�� @'����\��
=��
=C��                                    Bx�߭�  �          @�@I����{���R�N{C���@I����G��J=q�{C�XR                                    Bx�߼D  �          @�33@G���=q=u?z�C�'�@G���G�?�\@��HC�=q                                    Bx����  "          @��R@Mp���?�G�A'�C��{@Mp����?���AqG�C�U�                                    Bx��ِ  �          @�\)@z�H���?�A\z�C�e@z�H���\?�=qA�G�C��3                                    Bx���6  �          @���@y������8Q���C��=@y�����>�  @=qC���                                    Bx����  �          @��@�  ��Q쿂�\�C��f@�  ���H�z�����C�g�                                    Bx���  �          @˅@�33��\)�����C���@�33��=q�+����HC�c�                                    Bx��(  �          @��@�\)���ÿ�=q��C�(�@�\)���
�(������C��H                                    Bx��"�  �          @��@��H���
���R����C�@ @��H��G��˅�h��C���                                    Bx��1t  �          @���@z=q�Vff�dz��
�C��
@z=q�j�H�N�R���HC�h�                                    Bx��@  �          @�ff@h���I�������&�\C��\@h���b�\�}p��  C��                                    Bx��N�  "          @�p�@P���2�\��z��6��C��{@P���L(�����(=qC�Ф                                    Bx��]f  "          @�\)@n{��������6��C���@n{�����\�,G�C�C�                                    Bx��l  �          @�ff@e���ff��  �D�\C�O\@e��������H�;\)C��\                                    Bx��z�  T          @�=q@l�Ϳ�(���ff�Iz�C���@l�Ϳ����=q�A��C��                                    Bx���X  "          @�z�@p  �����G�
C�U�@p  �˅�����@�RC��=                                    Bx����  �          @Ǯ@c�
�z����B�RC��@c�
��R��\)�7�C��3                                    Bx�ত  �          @�G�@n{���z��=��C�T{@n{�\)��{�3(�C�{                                    Bx��J  �          @�ff@�  ���
��  �:�\C�xR@�  �����\)�9G�C��3                                    Bx����  �          @�=q@�\)��Q����2\)C���@�\)�=p���(��/��C�O\                                    Bx��Җ  �          @�  @��\���R��{�8��C�f@��\�������2ffC��                                    Bx���<  �          @ҏ\@���5����CQ�C�Ff@����\)���H�?(�C���                                    Bx����  �          @�G�@z�H?!G���33�J��A�@z�H>W
=��z��L�H@G
=                                    Bx����  �          @��H@�\)=�����p��2Q�?��@�\)��=q����2  C�L�                                    Bx��.  �          @�(�@��=u���\�+p�?333@��������=q�*��C�#�                                    Bx���  �          @�33@��;.{�����7��C��=@��Ϳ
=q�����6G�C�~�                                    Bx��*z  �          @�G�@�����������==qC��@�����Q���ff�8  C��R                                    Bx��9   �          @�ff@�z�\����-�HC��@�z�=p����
�+��C�|)                                    Bx��G�  �          @�p�@�\)>�����\�6z�?�z�@�\)�aG����\�6Q�C���                                    Bx��Vl  �          @�G�@��\�+�����8�C��\@��\��ff��  �5(�C���                                    Bx��e  �          @ڏ\@���У���Q��%p�C��
@����p����
�(�C��                                    Bx��s�  �          @ڏ\@��
��(����R�#33C�AH@��
��
�����RC���                                    Bx��^  �          @�33@���G��\)�33C�O\@���z��u��\)C��                                    Bx��  �          @�(�@��
��  �w��	C��R@��
��\�n�R��C��R                                    Bx�៪  �          @��@��H���H���\�=qC�u�@��H����z�H���C��                                    Bx��P  �          @Ϯ@�G����(��{C��@�G�����}p���C�q                                    Bx���  �          @�(�@����p�����G�C�
@�����y���  C��\                                    Bx��˜  �          @ʏ\@��Ϳ�33�u��ffC�� @������k���C�t{                                    Bx���B  �          @ə�@�{�����|���G�C�'�@�{�����vff�\)C���                                    Bx����  �          @��@�녿�����p��+{C�Q�@�녿����=q�%�RC���                                    Bx����  �          @�p�@u�c�
����H�HC���@u��(�����D��C�5�                                    Bx��4  �          @�=q@���
=�dz���C��@����^�R���C��=                                    Bx���  �          @��@�\)��  �p���ffC���@�\)��z��ff��  C��                                    Bx��#�  �          @�G�@��ÿ���C33��ffC��@��ÿ�G��=p���Q�C�H                                    Bx��2&  �          @Å@�=q���H�b�\��C��R@�=q��Q��\���	��C��=                                    Bx��@�  �          @��@�
=��=q�N�R��C�� @�
=����H����C��=                                    Bx��Or  �          @���@�=q���H�z���p�C�t{@�=q��{�����=qC�                                    Bx��^  �          @���@�
=����.�R�Ώ\C���@�
=�����(������C�                                      Bx��l�  �          @�\)@������2�\��G�C�L�@����(��-p���  C�k�                                    Bx��{d  �          @�=q@��R���ÿ������C�K�@��R��Q�������
C��)                                    Bx��
  �          @ə�@�p���\)�0�����
C���@�p�����)�����C��R                                    Bx�☰  �          @ə�@��ÿ����,���ʣ�C��@��ÿ\�'
=�ÅC�Q�                                    Bx��V  �          @�
=@�{��p��.�R��z�C���@�{����)�����C���                                    Bx���  �          @�z�@��\�Y���9����=qC�q�@��\���
�6ff��p�C��=                                    Bx��Ģ  �          @��H@�p���(��*�H��G�C��@�p���\)�$z���Q�C��                                    Bx���H  �          @���@�{�aG��W
=�\)C�0�@�{��=q�S�
��C�%                                    Bx����  �          @ƸR@��H?   ��(��AG�@���@��H>k������Bp�@J�H                                    Bx���  �          @�{@���>��H��G��@G�@�Q�@���>W
=����A\)@7
=                                    Bx���:  �          @�z�@QG�?�=q���V�A�ff@QG�?�ff��Q��[�RA���                                    Bx���  �          @�(�@0  @G���G��_�
BQ�@0  ?޸R�����g  B=q                                    Bx���  �          @�p�@4z�@.�R��  �I�
B033@4z�@{�����RQ�B$��                                    Bx��+,  �          @ȣ�@AG�@%����\�JG�B!�H@AG�@�
���R�R
=B��                                    Bx��9�  �          @�G�@a�?���  �Bp�A��H@a�?������H�G�A�p�                                    Bx��Hx  �          @���@fff@���G��7  A�  @fff?����z��<�
A�                                      Bx��W  �          @�\)@\(�@.{��G��*�RBQ�@\(�@ ����p��1�
Bff                                    Bx��e�  �          @��@W
=@����H�7\)B �H@W
=?�z���ff�=Q�A�R                                    Bx��tj  �          @�p�@g�?�\)��=q�7
=A��\@g�?�z���z��;p�A��\                                    Bx��  �          @�Q�@S�
?�����33�@��A�
=@S�
?�\)��p��Ez�A���                                    Bx�㑶  �          @�33@7
=?�����z��N��A��@7
=?�\)���R�T  A�z�                                    Bx��\  �          @��\@G�@\)��=q�\=qB2�@G�@G���p��c��B&ff                                    Bx��  �          @�?�p�?�����z�33B)
=?�p�?�=q���R��BG�                                    Bx�㽨  �          @ə�?�
=>aG�������@�R?�
=�u�����fC��                                    Bx���N  �          @Ϯ?�
=@�ff�k���B�u�?�
=@����y���\)B�8R                                    Bx����  �          @��
?�p�@�p��\(��ffB��?�p�@����g�� �HB��                                    Bx���  �          @��?�
=@hQ��n�R�1ffB���?�
=@^{�x���:��B�aH                                    Bx���@  �          @�p�?\)@W��w��B  B�G�?\)@Mp���  �K�\B�Q�                                    Bx���  �          @��R?c�
@   ��33�j�B��?c�
@z���ff�sz�B��                                    Bx���  �          @��;\)@r�\�g��.�
B�G��\)@h���qG��8(�B�u�                                    Bx��$2  �          @�p�?s33@hQ��E��z�B��3?s33@`  �N{�&G�B��3                                    Bx��2�  �          @�ff?�
=@x���G��ϮB�z�?�
=@s33�
�H��ffB��                                    Bx��A~  �          @���?�\)@w
=��  ���\B�� ?�\)@s�
��33��\)B��                                    Bx��P$  �          @�(����@g��ff�B�=q���@a���R�Q�B�u�                                    Bx��^�  �          @��>\@8���HQ��<=qB�33>\@1G��N�R�D�B���                                    Bx��mp  �          @9��?E�?������\
=B	�?E�>�׿���`�A�(�                                    Bx��|  �          @�  @J�H�*�H������=qC���@J�H�,(����R����C��R                                    Bx�䊼  �          @��\@�p��2�\?333@�(�C�޸@�p��1G�?L��A
�RC��)                                    Bx��b  �          @�(�@�Q��J=q>��R@`��C���@�Q��I��>�
=@�{C��                                    Bx��  �          @�33@�z��\��?O\)@��C���@�z��[�?k�A��C���                                    Bx�䶮  �          @�p�@��H�L��?�{A,��C�>�@��H�J�H?��HA<��C�e                                    Bx���T  �          @�=q@�p��9��?�=qAJ{C�XR@�p��7
=?�AW�
C��f                                    Bx����  �          @�{@����QG�?�G�Ab�RC�z�@����N{?�{Aq��C��=                                    Bx���  �          @�
=@����XQ�?��AF�\C��@����U?�
=AUG�C�B�                                    Bx���F  �          @�ff@��H�Vff?�z�A+�C�P�@��H�Tz�?�  A9�C�q�                                    Bx����  �          @�Q�@��R�{�?�A��HC�Q�@��R�xQ�@G�A��C���                                    Bx���  �          @ə�@�G��~�R?�  A�33C�^�@�G��{�?���A�33C��=                                    Bx��8  �          @�  @��z=q@�A�C�G�@��w
=@Q�A���C�y�                                    Bx��+�  �          @��
@����vff?�{AM�C�Ф@����tz�?���A\��C���                                    Bx��:�  �          @�G�@�(��fff?�\)ARffC��@�(��dz�?��HA`  C�33                                    Bx��I*  �          @��H@���]p�?���Ao
=C��=@���[�?�33A{�C��                                    Bx��W�  �          @��@�p��S�
?�  A�{C�U�@�p��QG�?���A�  C��                                     Bx��fv  �          @�33@���>{�u�&ffC�AH@���>{    =#�
C�@                                     Bx��u  �          @��R@����Dz����{\)C�@����Fff��p��p��C��                                    Bx���  �          @�=q@�p������G���ffC�q@�p����H�=p����C��                                    Bx��h  �          @���@�ff�\�����3�
C�7
@�ff��������/33C�)                                    Bx��  �          @�Q�@�
=�p������_�C���@�
=�\)����X(�C��{                                    Bx�寴  �          @��R@�  �U���G��"�HC���@�  �Vff�s33�G�C���                                    Bx��Z  �          @���@����Vff��z��:{C��q@����W������0��C��=                                    Bx���   �          @��@��H�HQ�O\)� ��C���@��H�H�ÿB�\��G�C��                                     Bx��ۦ  �          @�\)@�Q��J�H�=p�����C��3@�Q��K��333��{C�Ǯ                                    Bx���L  �          @�@�
=��Q쾞�R�B�\C��@�
=������z��5C��f                                    Bx����  �          @�ff@��R����=#�
>��C�J=@��R����=u?(�C�K�                                    Bx���  �          @�ff@���!녿+���
=C�E@���"�\�#�
���C�:�                                    Bx��>  �          @�\)@���7��8Q����C�p�@���8Q�.{�ҏ\C�g�                                    Bx��$�  �          @��R@�p��QG��E���C�,�@�p��Q녿:�H��\C�#�                                    Bx��3�  �          @���@�Q����k��\)C�0�@�Q��(��W
=��C�.                                    Bx��B0  �          @���@���   ?&ff@ə�C�b�@���   ?+�@У�C�h�                                    Bx��P�  �          @�=q@������?�Q�A5G�C�h�@����Q�?���A8  C�t{                                    Bx��_|  �          @\@�z��G�>�z�@4z�C�Y�@�z��  >���@<��C�\)                                    Bx��n"  �          @�@�33>.{�(����R?˅@�33>.{�(���
=?�ff                                    Bx��|�  T          @�G�@���ff���
�8Q�C�k�@���ff�u�
=C�k�                                    Bx��n  �          @��
@�\)�W
=��R��z�C�  @�\)�W
=�����
=C��)                                    Bx��  �          @�p�@�ff�$zῠ  �;33C�aH@�ff�%����R�9p�C�Z�                                    Bx�樺  �          @���@�(��B�\>.{?�C���@�(��B�\>8Q�?�\C���                                    Bx��`  �          @�=q@��
�K���Q��]p�C��@��
�K���Q��XQ�C�R                                    Bx���  �          @��@���n�R?��A+33C��@���n{?�33A,(�C���                                    Bx��Ԭ  �          @\@���|��?��HA��C���@���|��?�(�A�  C���                                    Bx���R  �          @�@O\)���@��A�\)C��
@O\)���@��A�p�C��
                                    Bx����  �          @�Q�?�
=���@p  B"�
C���?�
=���
@p  B"�C��R                                    Bx�� �  �          @�  ?��R��Q�@9��A�G�C�~�?��R����@8��A���C�|)                                    Bx��D  �          @�33@��\(�?��
At��C��@��\��?��
As�C��                                    Bx���  �          @�(�@y���z=q?�  A���C�t{@y���z�H?޸RA��C�o\                                    Bx��,�  T          @��@qG��vff@�RA�{C�5�@qG��vff@{A���C�,�                                    Bx��;6  �          @�{@n{���@A�G�C�3@n{���
@z�A��C�
=                                    Bx��I�  �          @�@s�
��(�?���A��
C�S3@s�
��z�?�ffA�  C�J=                                    Bx��X�  �          @�{@���p  ?�
=A�  C�9�@���qG�?�33A�  C�/\                                    Bx��g(  �          @�@�z��g
=?�z�A�  C�E@�z��g�?У�A�C�9�                                    Bx��u�  �          @�ff@�{�g
=@
=qA�z�C��H@�{�hQ�@Q�A�{C��\                                    Bx��t  �          @���@~{�j=q@A�\)C���@~{�k�@�
A���C��
                                    Bx��  �          @�33@dz��w
=@$z�A���C�b�@dz��xQ�@"�\A͙�C�K�                                    Bx���  �          @��@`���q�@%AՅC�q�@`���s�
@#�
A�  C�XR                                    Bx��f  �          @���@tz��g�@��A�
=C�K�@tz��h��@A��C�0�                                    Bx��  �          @�(�@i���]p�@;�A�(�C�G�@i���_\)@8��A�Q�C�"�                                    Bx��Ͳ  �          @��@L���u@I��B ��C��\@L���xQ�@FffA�
=C���                                    Bx���X  �          @��@c�
�k�@7�A�ffC�
=@c�
�n{@4z�A��
C���                                    Bx����  �          @�33@dz��fff@8Q�A��C�aH@dz��h��@5�A��
C�7
                                    Bx����  �          @�ff@���N{@z�A�
=C��=@���P  @G�A��HC��                                    Bx��J  �          @�G�@���:�H?�A��RC��@���<��?�\)A���C���                                    Bx���  �          @\@��
�4z�@�A��C��R@��
�7
=@�A�  C�˅                                    Bx��%�  �          @���@�33�@  @z�A�\)C�p�@�33�B�\@G�A��HC�C�                                    Bx��4<  �          @��R@�
=�;�@0��A�{C���@�
=�>�R@-p�A�33C�^�                                    Bx��B�  �          @��H@�z��
=@Q�B
(�C�%@�z���H@N�RB�C�Ф                                    Bx��Q�  �          @�33@�ff�-p�@@��A�Q�C��@�ff�1�@<��A�G�C�C�                                    Bx��`.  �          @��R@����G�@ffA�z�C��@����J=q@�A�
=C���                                    Bx��n�  �          @Å@��H�4z�@c33B�C��@��H�9��@_\)BG�C�Q�                                    Bx��}z  �          @�p�@����1G�@0��A�(�C���@����5�@,(�A���C���                                    Bx��   �          @�\)@�p��.�R@C�
A�  C�Ф@�p��333@?\)A�\C�~�                                    Bx���  �          @ƸR@���(Q�@H��A�G�C�B�@���-p�@Dz�A�C��=                                    Bx��l  �          @�ff@�=q�'�@P��A��C�q@�=q�,��@L(�A�(�C��)                                    Bx��  �          @�z�@��R��@7�A�G�C�#�@��R�  @3�
A�z�C���                                    Bx��Ƹ  �          @�@����@hQ�BG�C�s3@���!G�@dz�BG�C���                                    Bx���^  �          @Å@�ff�Q�@x��B\)C��\@�ff��R@uBz�C��q                                    Bx���  �          @���@|(���@�p�B/�C���@|(�����@��
B,Q�C���                                    Bx���  �          @�(�@����@Z=qB	=qC���@���G�@VffB��C�*=                                    Bx��P  �          @�@����z�@-p�AУ�C��{@������@(��A���C�5�                                    Bx���  �          @ƸR@����@��A�=qC�aH@��� ��@Q�A�(�C��                                    Bx���  �          @���@�{�Q�@��A��C��
@�{���@��A�(�C���                                    Bx��-B  �          @�Q�@�\)���@7
=A�p�C�R@�\)���@333A�Q�C���                                    Bx��;�  T          @���@�\)��Q�@S�
B�C�R@�\)��ff@P��B
(�C�xR                                    Bx��J�  �          @�{@�(��ff@�RA�Q�C���@�(���H@	��A�G�C�5�                                    Bx��Y4  �          @�  @���
�H@   A�=qC���@���\)?�
=A��C�@                                     Bx��g�  �          @�ff@�
=�	��?��HA��HC�(�@�
=���?���Aw�C��f                                    Bx��v�  �          @�@�33��Q�?�@�z�C�]q@�33���H?�\@��RC�C�                                    Bx��&  �          @���@�zῊ=q=���?��C���@�zΉ�=u?#�
C���                                    Bx���  �          @��
@�  �����z�����C��{@�  �
=q��
=��33C�^�                                    Bx��r  �          @�ff?�  ?L����z��=BQ�?�  ?z�H���  B�                                    Bx��  �          @��?У�?�{������BC33?У�@z����R�|BN��                                    Bx�鿾  �          @�{?��@z����(�B�\?��@!������y��B��R                                    Bx���d  �          @�(�?:�H@�
��33B�B��f?:�H@!���  �}��B��R                                    Bx���
  �          @�{>�z�@2�\��
=�w(�B�=q>�z�@@  ��33�m�RB�{                                    Bx���  �          @�\)>u@C33���
�l�B�{>u@QG�����b�HB���                                    Bx���V  "          @�=q>�(�@QG������cz�B��>�(�@^�R����Y�RB�
=                                    Bx���  �          @��H>\)@X����Q��_�B�ff>\)@g
=����U�B��                                    Bx���  �          @�>�  @b�\����TB�
=>�  @o\)���\�J��B�z�                                    Bx��&H  �          @�p���\)@hQ���(��O�B����\)@u���R�D��B�8R                                    Bx��4�  �          @��þ8Q�@j�H��  �P�B����8Q�@x����=q�FffB�L�                                    Bx��C�  �          @��׽�\)@�  �}p��%ffB��f��\)@��p  ��B���                                    Bx��R:  �          @���(�@�  �tz��)��B��῜(�@�p��hQ��\)BԞ�                                    Bx��`�  �          @�(��5@y���qG��.\)BȔ{�5@��\�e��#z�Bǳ3                                    Bx��o�  �          @�33?�@c33�����G�HB�Q�?�@p�����H�<��B�8R                                    Bx��~,  �          @���?��@E����
�Q�
B�?��@S33��
=�G\)B�                                    Bx���  �          @�ff?�\)@p���mp��+\)B��?�\)@|(��`��� p�B���                                    Bx��x  �          @���?333@��
�ff��33B��
?333@�\)�����  B�=q                                    Bx��  �          @�p�?n{@��H�(����B�=q?n{@��R��R�ڸRB���                                    Bx���  �          @��?���@r�\��=q��G�B�W
?���@xQ�У���  B�
=                                    Bx���j  �          @�
=?�z�@\)��(���G�B���?�z�@�����G��W�B��                                    Bx���  �          @��?Tz�@{��p��q��B�Ǯ?Tz�@(���G��e��B�B�                                    Bx���  �          @�=q�
=q�xQ�u�#  Cp�f�
=q�k���G��-�Co}q                                    Bx���\  �          @���(Q�����?8Q�@�z�Co�=�(Q�����>��@�(�Co�R                                    Bx��  �          @�ff�*=q����aG����Cn�)�*=q���������H��CnG�                                    Bx���  �          @�
=�7
=��  >�?�(�Cjp��7
=��  ���Ϳ�Q�Cjp�                                    Bx��N  �          @�33����녿#�
��
=Cu{�����׿fff�#�Ct޸                                    Bx��-�  �          @����G����Ϳ8Q���z�Ci
=�G���33�u�(��Ch��                                    Bx��<�  �          @��\�1��|(������(�Cj�{�1��s33����=qCi޸                                    Bx��K@  �          @i����{�>�R��(���{Cuc׿�{�8Q��33��
=Ct��                                    Bx��Y�  �          @����\�9�����	�Cn�=��\�0  � ����RCm33                                    Bx��h�  �          @����{�.�R����\Cc�f�{�%����Cbh�                                    Bx��w2  �          @Q녿����  �(��$C`Q�����{�33�.�C]�                                    Bx���  �          @�����G��1G�������Cm�\��G��(Q���
���Cl33                                    Bx��~  �          @�\)����=q�qG��F�Cdc�����
=q�z�H�PCaT{                                    Bx��$  �          @�
=�!��'
=�o\)�7Q�Ca�H�!��
=�y���A��C_
=                                    Bx���  �          @�z��33����q��O(�C[���33����y���X=qCW��                                    Bx���p  �          @�G�    �S33��ff��C��    �L�Ϳ�G����C��                                    Bx���  �          @@�׿z�H��Q쿙�����Cs(��z�H��{��=q�{Cr!H                                    Bx��ݼ  �          @*�H���8Q��\)���CH����(������CE��                                    Bx���b  �          @U�\)��33���#ffCO(��\)��  �  �*�CL
=                                    Bx���  �          @Tz���H�p���G���CI(���H�L�������CF5�                                    Bx��	�  �          @U�0  ���
������  CH�\�0  �k���z���\CFxR                                    Bx��T  �          @_\)�/\)�0��� �����CB&f�/\)����33��C?T{                                    Bx��&�  �          @vff�6ff��(��{�"��C<��6ff��  �   �$�\C9
=                                    Bx��5�  �          @��
���H� �������p�CS�{���H�ff����(�CQٚ                                    Bx��DF  �          @|(��(���:=q�+���Cc�q�(���6ff�^�R�J�\Cc0�                                    Bx��R�  �          @���W
=�^�R��{�z�\Ca�3�W
=�W���{��
=Ca�                                    Bx��a�  �          @�=q�W����
��������Ci�f�W����R�����Ch��                                    Bx��p8  �          @��
�1���\)�!���Co�
�1������7����HCn�                                     Bx��~�  �          @�=q��=q�3�
�����fffCl�ÿ�=q�=q��\)�tp�Ch�\                                    Bx�썄  �          @�
=��ff�������5
=C�3��ff�y����
=�EC~�q                                    Bx��*  �          @������\)����%(�C� ��������z��6  C~�3                                    Bx���  �          @�p��!���  �qG��z�Cp�f�!���ff��33��Cn�                                    Bx��v  �          @�G�������\��G��  CtT{�����Q���z��)�HCr��                                    Bx���  �          @�(��{��{���R�*�Co� �{�u������9\)Cm33                                    Bx����  �          @�  �(����
=���\�*z�Cm�R�(���vff�����9��Ck�=                                    Bx���h  �          @����=q��G������{CsQ��=q��{��(��*33Cq��                                    Bx���  �          @�=q��Q���G��@�����C|=q��Q������_\)��  C{�                                     Bx���  T          @�\)���
��  �   ��Q�C~&f���
�����@  ���
C}��                                    Bx��Z  �          @ᙚ��{��G��aG���RC|.��{��\)�~�R�
{C{B�                                    Bx��    �          @����
����1G�����C�lͿ��
����R�\���C�>�                                    Bx��.�  �          @�녿L���ə��AG����C�aH�L�������a���RC�8R                                    Bx��=L  �          @�z�����z��HQ��Џ\C��f�����33�i������C��                                    Bx��K�  >          AQ�^�R��\�p  ��p�C��ÿ^�R�����
����C��                                    Bx��Z�  �          A���\)� z���  ��Q�C�%��\)��z�����G�C��                                    Bx��i>  �          A(��n{�����R��z�C��3�n{����������C���                                    Bx��w�  �          A���=q�p��\������C��R��=q�(���
=���
C���                                    Bx�톊  �          AG���(��{�%��tQ�C�LͿ�(����XQ���p�C�1�                                    Bx��0  �          A\)�5��Q��ff�HQ�C|���5�����8Q���=qC|33                                    Bx����  �          A���=q���\)�m��C
�=q���QG�����C~��                                    Bx���|  �          A%�N�R���}p���{Cz8R�N�R�	�������(�Cyh�                                    Bx���"  �          A%��,����R�Q����C}���,���p���33��
=C}p�                                    Bx����  �          A(Q��J=q�{�U��33C{\)�J=q�����p���{Cz�R                                    Bx���n  �          A&=q���\����6ff�~�HCuٚ���\����i�����Cu#�                                    Bx���  �          A"=q��G���\�   �5Cm�
��G��
=�0  �|  Cm(�                                    Bx����  �          A%������  ��(��/�Cl�
�����z��/\)�uCl(�                                    Bx��
`  �          A&�H��  ����0  �uG�CnaH��  �  �aG���ffCm�                                     Bx��  �          A!���z���G���
�UG�Ch����z���G��A���ffCh�                                    Bx��'�  T          A �����������#33�n=qCh��������(��P����
=Cg��                                    Bx��6R  �          A$z����H����z���CmO\���H��Q����
��Ck޸                                    Bx��D�  �          A$����33�p���(����Crs3��33��p��������Cq33                                    Bx��S�  �          A%�h����
��p��֣�Cv���h��� (����R���\Cu��                                    Bx��bD  �          A*{�`���Q������ÅCx���`����������=qCw��                                    Bx��p�  |          A,z��N�R�{�aG����HC{p��N�R�  ���£�Cz��                                    Bx���  �          A+
=�6ff���!G��Z�RC~\�6ff��H�^�R��{C}��                                    Bx��6  �          A'��6ff�(��p��D  C}�3�6ff�  �I������C}G�                                    Bx���  �          A&{�?\)�p��\(���G�C}{�?\)�\)�����"�\C|�)                                    Bx�  �          A&{��  ��޸R��Cv޸��  �=q�*=q�n�RCvaH                                    Bx��(  �          A'��mp�����{�ffCx�=�mp���#33�c33Cx^�                                    Bx����  �          A(���9��� Q쿗
=��ffC}�H�9������
�H�?
=C}��                                    Bx���t  �          A%G��,����H�c�
���\C~c��,����׿�{�)�C~+�                                    Bx���  �          A���R�\��(�@�  B  Cts3�R�\��p�@���A��Cv)                                    Bx����  �          A z��[����@S�
A�Q�Cx���[��p�@��A]CyQ�                                    Bx��f  �          A&{�e��?h��@��HCy�{�e���\��\)�ǮCy��                                    Bx��  |          A2ff��\�
=�<(�����C�/\��\�p��x����z�C�)                                    Bx�� �  T          A<zᾣ�
�������%�C��3���
��
=�	�<C��
                                    Bx��/X  �          A=G��(���0Q�������Q�C�G��(���((���p���C�33                                    Bx��=�  �          A=p�?(��(��ۅ�  C��3?(��  ��=q�&�HC��)                                    Bx��L�  �          A=G�>�������\���C���>���33���7�
C��                                    Bx��[J  �          A<�Ϳ+��G�����3ffC��ÿ+�����=q�Kz�C�l�                                    Bx��i�  @          A=���W
=��H�Q��8��C�B��W
=��\)�G��P�C�*=                                    Bx��x�  |          A6=q�z����H��H�8�\C��H�z������P�C���                                    Bx��<  �          A9p�@
�H� ��������=qC��@
�H��\���H�{C�~�                                    Bx���  T          A=�@=p��\)������C��@=p���R��\)�#z�C���                                    Bx�龜  �          A<  @z��33��=q�{C���@z��
=��� �C�xR                                    Bx��.  @          A6=q?�G��
ff���
���C�?�G���=q� Q��4  C�Y�                                    Bx����  |          A0Q���H�����H���C��R���H�(���33��\C���                                    Bx���z  T          A3�������������C�� �����\������C��f                                    Bx���   6          A5G�=�Q��Q���Q��(��C�L�=�Q������{�B=qC�U�                                    Bx����  �          A4�Ϳ�33�"�\�y����\)C����33��R��  ��33C�o\                                    Bx���l  h          A4  ��
=�%p��j�H���C��R��
=��������  C�xR                                    Bx��  �          A8�׿u�#33��(��ˮC�N�u���������RC�"�                                    Bx���  
�          A>�R?�z��������$��C���?�z���z���
�=��C�                                    Bx��(^  T          A@��@*�H��33�,(�C��q@*�H��(��G��D��C��3                                    Bx��7  T          A=�>����	����,Q�C�O\>��������  �FG�C�z�                                    Bx��E�  �          A=p�@G���
��
=�)��C���@G������{�B�HC���                                    Bx��TP  |          A;
=@\)���z��(�C��@\)���{�.�C��R                                    Bx��b�  T          AB�\@Q��33��
�*�\C�޸@Q���{��H�D
=C���                                    Bx��q�  �          A6�R@�H��H����(�HC�9�@�H��\)�
{�B{C�:�                                    Bx���B  �          A&�\@b�\=������
�\?���@b�\?�G�����=A~ff                                    Bx����  |          A@Q�?У��p���=q��C���?У���H��0�HC�w
                                    Bx��  �          A9�@z��	p�����p�C���@z����(��8ffC���                                    Bx��4  �          A1�@W�����{�{C��
@W������(��&�
C��                                    Bx���  �          A333@��H��{���H�  C��)@��H�ڏ\��ff�.(�C�9�                                    Bx��ɀ  �          A5�@�
=���
����1z�C�f@�
=��������GffC�.                                    Bx���&            A9G�@�����G������+��C���@����e����9�\C�|)                                    Bx����  T          A;�@�p����z��Fp�C�@ @�p���33���L�
C�Ff                                    Bx���r  �          A?33@���љ�����7�\C��R@������(��L�\C�Y�                                    Bx��  |          A6�R@�ff�u���z��"��C���@�ff�:�H��G��-��C��H                                    Bx���  �          A6�\@ᙚ�����
=�$z�C��
@ᙚ�z�H�   �3{C�s3                                    Bx��!d  �          A5G�@θR��33���%C���@θR��z���R�7  C�#�                                    Bx��0
  "          A4��@�����������0�C��R@��������
�E�C�z�                                    Bx��>�  |          A6ff@߮��(������C��3@߮��Q���\)�${C�,�                                    Bx��MV  �          A;\)@��H��33��p��'C��@��H��(���
�7p�C�s3                                    Bx��[�  �          A6ff@�ff���\�����.�C���@�ff���\�=q�?�
C��f                                    Bx��j�  �          A4��@����;��(��P�C�G�@��Ϳ�=q����Z�C���                                    Bx��yH  �          A1G�@���>Ǯ����G�@��
@���?˅��H�y�\A���                                    Bx���  �          A2{@Q�@"�\�(Q�ffB;�R@Q�@u��"{
=Bh��                                    Bx��  �          A0z�@�@g
=�#33��Br�@�@�33��R�u�B��=                                    Bx��:  �          A/�@�����
=���r�
C���@���?B�\�33�q�HA\)                                    Bx���  "          A2=q@�z��  ���MffC�N@�z῕�
�H�T�
C��                                    Bx��  �          A/
=A�?z��У���@tz�A�?�33������A��                                    Bx���,  T          A9p�A*ff�޸R�hQ����HC�\)A*ff���\�tz�����C���                                    Bx����  
�          A333A{�����=q�˙�C�9�A{�n{���\���C��=                                    Bx���x  "          A>�HA"{��ff���
��(�C���A"{��z���{��p�C��                                     Bx���  �          A:{A���33�p  ����C���A����H������(�C��)                                    Bx���  �          A8(�A�\��Q��Tz����\C�.A�\��G��y������C�J=                                    Bx��j            A6�HA#����H�333�dz�C�
A#��l���S33��33C��                                    Bx��)  �          A7\)A&�R��G���
�;33C�j=A&�R�mp��3�
�dQ�C�5�                                    Bx��7�  "          A7�
A(���r�\�   �IG�C��A(���\(��>{�o�C���                                    Bx��F\  
�          A4z�A$z�����޸R��RC���A$z��y������:{C���                                    Bx��U  "          A/�
A (���p����
�p�C��3A (��y����
�A�C�Z�                                    Bx��c�  �          A/�
A�H��p��˅��C���A�H���	���4��C��
                                    Bx��rN  �          A4��A#\)�}p��5�h��C�ffA#\)�c33�U��33C�e                                    Bx���  
�          A9��A$������Tz����C�AHA$���e�tz���ffC�ff                                    Bx��            A@��A*�\���R�qG����C�8RA*�\�k���G����\C�y�                                    Bx��@  
�          A:=qA,���p�׿�\)���C�h�A,���^�R�ff�;33C�3                                    Bx���  �          A=��A0���c33�Q��8��C��A0���Mp��4z��\z�C��=                                    Bx��  T          A>ffA4Q��@  ����9��C��A4Q��)���1G��W33C�`                                     Bx���2  �          AB=qA5p��O\)�8Q��[\)C�fA5p��5��Q��z�HC��q                                    Bx����  
�          AAG�A4  �S�
�&ff�G�C�˅A4  �<(��@���hz�C��                                   Bx���~  �          AB{A8z��^�R���\��Q�C��)A8z��Q녿�(���RC��                                    Bx���$  �          AL  AA��P  �!G��6=qC�w
AA��8Q��;��Tz�C�G�                                    Bx���  7          AH��A;�
��  �}p���G�C�
=A;�
���\��ff���
C�h�                                    Bx��p  �          AF�RA:ff��=q�����
=C�]qA:ff�w
=��=q�\)C��{                                    Bx��"  �          AG
=A<���U��(��C��A<���B�\����2{C��f                                    Bx��0�  
(          ABffA9���6ff�{�(��C�
A9���!��%�D��C�ٚ                                    Bx��?b  �          A?�A7\)�1G���\���C�5�A7\)�{����8��C���                                    Bx��N  
�          A?�A7�
�(Q��z��\)C��\A7�
�z��=q�9��C�J=                                    Bx��\�  �          A?�A8Q��ff�G��.�HC�>�A8Q��G��$z��F=qC��                                    Bx��kT  
�          A>=qA8  ����\)���C�EA8  �33�
�H�(Q�C��\                                    Bx��y�  "          AA�A<���ff�
=�5�C�^�A<���  �h�����\C��)                                    Bx��  
�          AAp�A4  ��G��z=q����C��=A4  ���H������
C��                                    Bx��F  
�          AAA-�����H������C�,�A-���=p������=qC�\                                    Bx���  
�          AB=qA,(���=q�����\)C�  A,(����
���H��p�C�'�                                    Bx��  T          A>�RA#�
��������RC�eA#�
>�=q���H���?��R                                    Bx���8  
�          A=G�A   �0���ƸR��=qC�fA   >.{��  ���?}p�                                    Bx����  �          A9p�A�R���������HC��RA�R?8Q���(����H@�                                    Bx����  
�          A6�HAG�<��
������(�=�AG�?Q������\)@�G�                                    Bx���*  �          A7�AG�=�\)��z�����>\AG�?aG����\��z�@�z�                                    Bx����  
�          A4��A�?&ff��Q����@qG�A�?�z����
��G�A�\                                    Bx��v  i          A3
=A)p�>�Q�n{��G�?�
=A)p�>��aG���
=@'
=                                    Bx��  T          A<(�A'33@1G���G���ffAm��A'33@XQ������=qA�\)                                    Bx��)�  �          A:ffA2�\@'
=������\AR�HA2�\@7
=�Ǯ��\)Af{                                    Bx��8h  �          A8(�A+33@l��� ���!p�A��\A+33@|(���(����
A��                                    Bx��G  �          A2�\A=q@U�������A��\A=q@y���s33��ffA�ff                                    Bx��U�  T          A*�\A��@tz��������A���A��@�  ���R��=qA�                                      Bx��dZ  @          A(��A33@�
=�#�
�dz�BG�A33@�  >8Q�?�  B�H                                    Bx��s   h          A%p�A\)@���=u>�33A�(�A\)@�
=?E�@���A�                                    Bx��            A((�A z�@zῨ������AP��A z�@\)�xQ���(�A^�H                                    Bx���L  �          A*{A%��@�׾�p�� ��AE�A%��@�\�L�;�  AG�                                    Bx����  T          A(��A$(�@��=#�
>L��AR�RA$(�@�>\@33AP(�                                    Bx����  
�          A&�RA#
=?�p����R�ۅA�
A#
=?�녿��\���
A�                                    Bx���>  
�          A#�A  >B�\�0  �|  ?��A  ?z��,(��v�R@X��                                    Bx����  �          A"{A��u�����z�C��
A�>������?=p�                                    Bx��ي  
�          A!G�Ap�?+��P����\)@�=qAp�?����HQ����R@��                                    Bx���0  #          A@��ÿn{��\�o�
C��H@���>�z��\)�r  @^{                                    Bx����  �          A=q@|(���
=���}�\C��@|(��W
=��
�C�z�                                    Bx��|  
w          A�@%�ff��ǮC�z�@%�L���G��qC�e                                    Bx��"  7          A=q@1녿��H���aHC�7
@1녾\�\)�C�\                                    Bx��"�  �          AQ�@���Q����\C���@��(���� C��3                                    Bx��1n  
�          A\)@.{����=q#�C�q�@.{��Q��  ffC�                                    Bx��@  
�          A��@0  ��z��
�\Q�C�h�@0  �Ǯ���
=C��{                                    Bx��N�  "          A(�@�H���
=qffC���@�H���������C�H�                                    Bx��]`  
�          Ap�@8Q쾅��Q�z�C�w
@8Q�?��\�\)W
A�{                                    Bx��l  ?          A{@��
?�Q��Ӆ�7G�A�ff@��
@8Q���  �+
=A�G�                                    Bx��z�  7          Aff@��@:=q��ff�K{A���@��@z�H��ff�8z�B{                                    Bx���R  
�          A��@��H@  ����z�A��@��H@E���Q��G�A�=q                                    Bx����  J          A�@��@'
=����홚A��@��@P����\)��A��H                                    Bx����  
�          A
=A�
?����R��\)AF{A�
@#33��z���(�A���                                    Bx���D  
�          A�R@�p�@�\��G��p�Ag
=@�p�@5������A�G�                                    Bx����  T          A\)@��@C33��z��33A���@��@xQ���(���A��H                                    Bx��Ґ  
�          A�R@��\@W
=��  �<Q�A��\@��\@�33���)�B�
                                    Bx���6  i          Ap�@Ϯ@�ff���\��B�\@Ϯ@�=q�u��ffBQ�                                    Bx����  J          A�@��@w
=�7���
=A�(�@��@���G��g33A��
                                    Bx����  "          AQ�A(�@��\��z��;�
A�z�A(�@�=q�����z�A���                                    Bx��(  +          A��A�R@\)?��\@�
=A�(�A�R@p��?���A9AŅ                                    Bx���  i          A�@�p�@�  @z=qA��B  @�p�@w�@���A�
=A��\                                    Bx��*t  �          A�@Mp�?\(��,���G�Ao33@Mp�?�  �"�\��A�                                    Bx��9  �          A=q?�{�z���H�{C�%?�{��ff�
�H��C�                                      Bx��G�  �          Ap�@n�R��33�����{ffC���@n�R�k���z�L�C�>�                                    Bx��Vf  "          A�@`��<#�
��=q��>k�@`��?���\)�)A�                                      Bx��e            A(�@�{@���&ff���B�
@�{@���=���?#�
B                                    Bx��s�  �          A�@�z�@���}p���z�B\)@�z�@�  �W
=����B=q                                    Bx���X  �          A33?�G�>.{�G�¢��@�(�?�G�?�z��\)�B@�\                                    Bx����  �          A�?Y��>\�(�¨.A�?Y��?�(�����\B~Q�                                    Bx����  �          A�H?�?&ff��©�fBL��?�@���R33B�W
                                    Bx���J  �          A�\?�ff?�\�p�8RBB?�ff@Dz����HB|�\                                    Bx����  �          A��?���@2�\���Bap�?���@�(��33�z  B���                                    Bx��˖            A"�\>�=q@  �{(�B�{>�=q@l����H\B��                                    Bx���<  �          A�
?��H>W
=�p�¥�qA (�?��H?�
=�
=�)BX��                                    Bx����  �          A Q�?B�\�u�33«L�C�Ф?B�\?�
=���¡Q�Bx��                                    Bx����  T          A!녽�?xQ�� ��¨�)B�z��@p��z��B��q                                    Bx��.  T          A�
��ff?������=B�8R��ff@QG���
B�B�aH                                    Bx���  �          A �ü#�
?�33�{£�B���#�
@7
=����B�z�                                    Bx��#z  �          A$��@5�?�  ����A�
=@5�@-p��  #�B.��                                    Bx��2   �          A#
=@XQ�@�R��\B��@XQ�@fff�{�s�B;\)                                    Bx��@�  �          A�\@^�R@0���	��zffB�\@^�R@�Q�� ���b\)BDG�                                    Bx��Ol  �          A  @tz�@P����l{B"
=@tz�@�\)�����Sz�BFQ�                                    Bx��^  �          Ap�@�
?�(��33�=B/
=@�
@U�����Bi(�                                    Bx��l�  �          A��@��
@�H�Q��r�HA�\)@��
@k�� ���^\)B&�                                    Bx��{^  �          A@��@
�H�\)�x�A��H@��@\���z��e
=B!�                                    Bx���  �          A   @�\)@���wz�Aң�@�\)@Y���ff�d�B(�                                    Bx����  T          A$Q�#�
?����"ff£�3B��#�
@>{���W
B�aH                                    Bx���P  T          A"ff=���?���� z�¦Q�B���=���@,����  B��
                                    Bx����  �          A�H?Y��?�ff�
=��B�p�?Y��@O\)���ffB���                                    Bx��Ĝ  �          A   =L��@ff��
��B�(�=L��@c33��B�B�Q�                                    Bx���B  �          A#�>�@'
=�G��qB�{>�@�=q���=B�z�                                    Bx����  �          A"ff?0��?�33�33�B��\?0��@HQ��p�B�.                                    Bx����  �          A��?��H=�����¥�q@��H?��H?�������BQ{                                    Bx���4  �          A ��?��?��� 
=BC�
?��@2�\���B��                                     Bx���  �          A&�\�Q�@u��H\B�#��Q�@�
=��
�d��B�u�                                    Bx���  �          A$(��G�@�G���
�Y(�B�{�G�@�\)��z��:�B�B�                                    Bx��+&  �          A �ͽ��
?����p�� B��)���
@P  �33�B��R                                    Bx��9�  �          A!p�@ �׾�����p�C�Q�@ ��?���Q��A��
                                    Bx��Hr  �          A%p�?�=q>��H�"=q��Apz�?�=q@G��33L�B?G�                                    Bx��W  
�          A%G�@   ?=p��!G��A���@   @������BB{                                    Bx��e�  �          A#�
@
�H@���R��B9��@
�H@n{���Bo                                      Bx��td  �          A&{@�?���!���=A���@�@%���k�BJQ�                                    Bx���
  �          A)?�?�G��$��  B�?�@C�
�\)�\Bl
=                                    Bx����  �          A*=q@��@_\)��u�Bg=q@��@�{��R�n
=B�
=                                    Bx���V  �          A)��?�@XQ��(�#�Bw�\?�@�=q�=q�rffB��                                    Bx����  �          A%�?�ff@�{�����B�Q�?�ff@�������`{B��q                                    Bx����  �          A%p�?���@���ff\B�Q�?���@�{�
�H�d�\B�33                                    Bx���H  �          A'�
?��@n{��
�3B�#�?��@�����op�B��q                                    Bx����  �          A&{?=p�@x���z��)B��{?=p�@�G��G��kG�B�(�                                    Bx���  �          A&�R@	��@j�H���3Bn��@	��@������h�B�\                                    Bx���:  �          A%�@�\@Tz��G��RBi�\@�\@�\)�\)�p\)B�Q�                                    Bx���  �          A&{?ٙ�@B�\�33z�Br�?ٙ�@�\)�{�y=qB�W
                                    Bx���  �          A&=q@G�@(Q�����{BQ�R@G�@��\�z��ffB~��                                    Bx��$,  �          A$��>��@J=q�\)B�{>��@��\�Q��|  B��                                    Bx��2�  �          A&{����@U��33�B�8R����@�Q��G��u
=B���                                    Bx��Ax  �          A'�
����@z��!��z�C5ÿ���@e���R�HB�                                    Bx��P  �          A'\)��
=?�\)�!G��RC�ÿ�
=@8Q��  =qB��                                    Bx��^�  �          A(��?�\@3�
�!� B�L�?�\@����8RB�k�                                    Bx��mj  �          A'33?
=@���"{
=B��?
=@qG���R�B�Q�                                    Bx��|  �          A'\)�aG�@Tz��G��Bў��aG�@����\)�x��B��
                                    Bx����  �          A((��L��@7
=� ���B�Q�L��@�33�  �B��H                                    Bx���\  �          A%p���{?�\)�!p���B�W
��{@XQ��
=��B�aH                                    Bx���  	u          A"ff���\>8Q����¨
=C)�q���\?��\)=qB��                                    Bx����  �          A �ÿ����  �Q��RCi5ÿ����  ��H¤��C>��                                    Bx���N  T          A\)��{>��R�Q�£ǮC'O\��{?��
�B���                                    Bx����  
�          A!����33>��R�G��fC*� ��33?����R��C�
                                    Bx���  
          A"{��\)�\��¡  CA��\)?����\aHC\                                    Bx���@  ?          A#33����G�� ��¢p�CRB����?@  � ��¢��C�f                                    Bx����  
�          A#\)�Tz�>��!©(�C� �Tz�@ ����RQ�B��)                                    Bx���  
�          A!�>�G�?�(��{�{B�#�>�G�@L���(�z�B�33                                    Bx��2  T          A'33?��@W��(�  B��?��@����=q�s��B�=q                                    Bx��+�  
�          A ��?Q�@+���aHB��?Q�@��H����B�Q�                                    Bx��:~  �          A$  ?aG�?���"=q¤�\BI�?aG�@%��(�B�W
                                    Bx��I$  
�          A   ?c�
?�����¡�RB^�
?c�
@1������B�ff                                    Bx��W�  
(          A�>��?�33���£W
B��>��@5��z�
=B���                                    Bx��fp  
�          A=q>�(�?\��
¡(�B��3>�(�@;���\(�B��                                     Bx��u  
�          A(�=�\)?����­z�B���=�\)@ ���z�.B��                                    Bx����  �          A��>�>u�z�°�qBtQ�>�?��H�{ǮB�
=                                    Bx���b  T          A
=>aG�?&ff�{«�)B��R>aG�@��H�)B��R                                    Bx���  T          A��?5?�33��  B���?5@Tz���R\)B��3                                    Bx����  
�          A!G�?Y��?�=q��)B�.?Y��@R�\���B�
=                                    Bx���T  T          Aff?�?�{�� 8RB�Q�?�@C33�{k�B�(�                                    Bx����  T          A!��?8Q�@'
=�ff�3B�Q�?8Q�@����ff�
B��
                                    Bx��۠  �          A ��?\(�@}p��ff8RB��{?\(�@�G��33�f�B��{                                    Bx���F  
�          A%�?(�?��R�#33¤�{B�?(�@0  �ff.B��                                    Bx����  
�          A&ff?E�?^�R�$z�§#�BB�?E�@��� ����B��                                     Bx���  �          A%�?u?�=q�#�£�fBB(�?u@&ff�33�)B��{                                    Bx��8  
Z          A'33?aG�?��H�$(�¡�Bk��?aG�@>�R��H=qB�
=                                    Bx��$�  T          A&ff>��?=p��%G�«L�B�8R>��@��!��p�B��                                    Bx��3�  �          A$�ÿ�  �E��#�¥��CY�
��  ?E��#�¥��Cc�                                    Bx��B*  T          A#�=�\)��\� ��   C�/\=�\)��  �#33±
=C�3                                    Bx��P�  T          A=q?�\�.{�33�qC�Q�?�\?��R�
=B��                                    Bx��_v  �          A Q�@ff?8Q���H8RA��@ff@	����ǮB7�                                    Bx��n  
�          A%�?�\)����#
=¤C���?�\)?�{�"=q B�B(�                                    Bx��|�            A&ff?�33��33�#�¡�C��\?�33>\�$Q�¦�A�                                    Bx���h  	�          A+�>��#�
�&=qB�C�ff>��}p��*ff©G�C��)                                    Bx���  �          A,��?�R�\���#\)
=C��?�R��33�)��C��                                    Bx����  "          A,��=#�
�Fff�%p���C�b�=#�
���
�+
=£��C��f                                    Bx���Z  T          A,z�>���a��"�R��C�33>�����R�)G���C�                                      Bx���   
�          A%�����w����C�3���=q� ���C���                                    Bx��Ԧ  �          A)p�����
=�
=�C�� ���/\)�#\)�C�8R                                    Bx���L  	�          A*=q?:�H��%G�Q�C��?:�H�J=q�(��¨aHC�l�                                    Bx����  �          A%�>��H�{�!����C�B�>��H�333�%�ªz�C�}q                                    Bx�� �  "          A&=q?Tz��!G�� (�B�C�3?Tzῂ�\�$Q�¥ffC�z�                                    Bx��>  
(          A!p���z��l(��G��qC��q��z����Q��\C�Z�                                    Bx���  
(          A �Ϳ}p�����=q�y�C��)�}p��Mp��\)�C|��                                    Bx��,�  
(          A!녿�����\)��u{C�w
�����XQ��33��C|�                                    Bx��;0  �          A$  �h����  ����x�C����h���W
=�
=z�C~�                                    Bx��I�  �          A�Ϳ��H��(����CeǮ���H��33�{¡�
CA��                                    Bx��X|  T          A (���Q��(Q��G�8RCq:ῸQ쿙���k�C[                                    Bx��g"  "          AG���(���  ���aHCy�\��(��*�H����3Cq(�                                    Bx��u�  �          A�
��Q���p���b
=C~h���Q��{��z���Cy�)                                    Bx���n  
�          AQ��������R�Z{C}(������
=�
�R�yCx��                                    Bx���  
�          A����
��Q���R�a�
C�/\���
�������
=C|c�                                    Bx����  �          A����  ��=q�ff�=C�ÿ�  �/\)�ff�Cy��                                    Bx���`  �          A{�xQ����R�=q.C��f�xQ��7��ff=qC{ff                                    Bx���  �          A�\�u��
=�
=�u�C��f�u�Z�H�����C���                                    Bx��ͬ  T          A(������xQ���RǮC��;����#33�=qQ�C�J=                                    Bx���R            A��>k��%�33u�C���>k���p���¤�C�9�                                    Bx����  
=          A��?������\)ǮC�E?��>L���z�£�A
=q                                    Bx����  �          A?��Ϳ:�H��¤(�C�+�?���?#�
��
¤�qA�{                                    Bx��D  T          A�?�������¢�C�|)?��?O\)�z�¡\)A�\)                                    Bx���  �          A�R?��\?���
=¢z�B6�
?��\@���
=� B��
                                    Bx��%�  "          A�H�޸R�p�����=qCPE�޸R>���33C+h�                                    Bx��46  "          A���������ҏ\�/
=Cg+������������G��CaQ�                                    Bx��B�  "          A33�����Q���Q��z�Ckn��������(��%�HCg�{                                    Bx��Q�  
�          A���  ���
��z��?Q�Cd����  �s�
�����V�RC]��                                    Bx��`(  
o          A��������  �Ϯ�'�CbB�����������p��=��C\aH                                    Bx��n�  
�          A  �������R�ȣ�� �Cf�������Q�����8�C`�                                    Bx��}t  
�          Ap�������p���(��7G�Ccc������x����  �N  C\�R                                    Bx���  
�          AG��J�H���������ZQ�Ck(��J�H�Y����s�RCc
=                                    