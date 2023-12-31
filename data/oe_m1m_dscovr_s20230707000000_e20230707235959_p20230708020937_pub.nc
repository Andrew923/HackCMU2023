CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230707000000_e20230707235959_p20230708020937_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-08T02:09:37.854Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-07T00:00:00.000Z   time_coverage_end         2023-07-07T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data           records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��_�  �          A\)�ƸR@�=q@g�A�33Cc��ƸR@��@z�HA��C�=                                    Bx��nf  
�          A�R��{@��@��
B��C)��{@��
@�B�\C��                                    Bx��}  "          A녿���@�@��
BFB��)����@���@�p�BS�B�                                      Bx�ڋ�  "          A  ��z�@��@���Bg��B��H��z�@��\@�z�Bs��Bӳ3                                    Bx�ښX  �          A�\��=q@���@vffA�  C �3��=q@���@�p�A�C                                    Bx�ڨ�  �          @��
����@%���ff�У�B�{����@*�H��z����B�ff                                    Bx�ڷ�  �          @���7
=@����S33��
=B����7
=@�ff�A���G�B�                                    Bx���J  �          A�����@Mp���Q��N�C������@e��=q�G  C�)                                    Bx����  �          AQ쿅�@
�H�	p��B�.���@(���\)��B�\                                    Bx���  �          A�?��
��ff��¡�C�Q�?��
    �G�¢�{=�                                    Bx���<  �          @�\)�W�@��@n{B33B�aH�W�@���@}p�B{B��f                                    Bx�� �  �          A���\)@�{@ÅB"
=B��
�\)@�33@�p�B,(�B���                                    Bx���  �          A��g
=@��R@���A�p�B�(��g
=@�{@���BB�                                    Bx��.  �          A�R�У�A�@\��A��
B�(��У�A��@z=qA�Bɣ�                                    Bx��,�  
�          A���Q�@��@ϮB%�C����Q�@��@�  B-�
C	�{                                    Bx��;z  
�          A�\��  @��H@�Q�B3�\C	k���  @�\)@߮B;�C�q                                    Bx��J   !          A�R��ff@p  @陚BM��Cٚ��ff@W�@�BUG�C�H                                    Bx��X�  #          A��~{@\)@�p�BW{C�)�~{@e�@��
B_Q�C	��                                    Bx��gl  
�          A
=�?\)@�  @�RBC��B���?\)@��@�  BN\)B�Q�                                    Bx��v  
[          Ap��HQ�@�G�@�G�BK33B�.�HQ�@���@���BU=qB�#�                                    Bx�ۄ�  
Z          A��vff@\(�@��Bc\)C
.�vff@B�\A�Bj�HC��                                    Bx�ۓ^  �          A�����@���@�z�A��RB��H����@���@�
=B��B�                                    Bx�ۢ  
�          A Q��j=qA(�@FffA�z�B�aH�j=qA	p�@a�A�B�.                                    Bx�۰�  "          A!p����HA�?�G�@���B�\���HA
ff?�Q�Az�B�k�                                    Bx�ۿP  
�          A#�
��=qA>\)?E�B�\��=qAp�?z�@P  B��                                    Bx����  T          A�
�"�\A
=�fff���BԊ=�"�\A�J�H���
B���                                    Bx��ܜ  T          A�ÿ�@����ff�,Q�B��
��@��H���H�!G�B��H                                    Bx���B  �          A"�\�=p�@أ���
=�>�RB�k��=p�@��
��z��3�\B���                                    Bx����  T          A#33��(�@���� (��Q�B��)��(�@����R�F��BɊ=                                    Bx���  "          A!G���(�@׮��=q�:�HḄ׿�(�@�\�߮�0{B�z�                                    Bx��4  "          A$z�?�\@��H��\)�-�B�
=?�\@�����
�#  B�aH                                    Bx��%�  T          A#�?��R@�{���1
=B�k�?��R@����Q��&Q�B�G�                                    Bx��4�  "          A ��?�33@�Q���\�C�RB��3?�33@ʏ\��G��9(�B�                                      Bx��C&  �          A�@ ��>Ǯ���8RA.ff@ ��?Q��Q��A�{                                    Bx��Q�  �          A"ff?�=q@j=q���k�B�?�=q@��\��\L�B�=q                                    Bx��`r  "          A��>aG�@�z���=q���B��>aG�@�(���\)�
=B���                                    Bx��o  �          AzῚ�H@mp���ff�p�B�8R���H@�=q�����up�B�33                                    Bx��}�  �          Az��$z�@�
=��
=��
=B܏\�$z�@����y����p�Bۙ�                                    Bx�܌d  �          AQ��ff@�\)?ǮA5�B�L��ff@��?��A\��BӞ�                                    Bx�ܛ
  �          A ����
?�A�B��C����
?���A=qB���C�                                    Bx�ܩ�  T          A��
=@�Ap�B��C33�
=@ ��A
=B���C��                                    Bx�ܸV  "          A�����@���@�  B�\B�z����@�ff@���B{B۞�                                    Bx����  !          A=q����AG�@uAɮBÅ����@���@�{A݅B��)                                    Bx��բ  
�          A���#�
@�
=@���A��HB��\�#�
@�Q�@��
B \)B��{                                    Bx���H  �          A?�  AQ�@Q�AqB�  ?�  A
�\@0��A�z�B��
                                    Bx����  �          A=q>k�AG�����Z=qB��>k�A�\���333B�#�                                    Bx���  "          Aff�L��A=q����E�B�33�L��A\)������B��                                    Bx��:  �          @��
����@�33�����=qBĊ=����@�z�=p���ffB�u�                                    Bx���  "          @�����\@�Q�>��@   B�녿��\@�?\)@��HB���                                    Bx��-�  �          @��
���@�?�=q@��
B�{���@�  ?��A"�\B�B�                                    Bx��<,  
(          A��%�@���?ٙ�A;�B����%�@�{@ ��A_
=B�#�                                    Bx��J�  �          A��8��@�{@
=qAeG�B����8��@�33@�RA�  B�ff                                    Bx��Yx  
�          A
=�5@��@G
=A�Q�B�(��5@�@Z=qA�p�B���                                    Bx��h  
�          AQ��L(�@�  @�ffA�33B�
=�L(�@ڏ\@�\)A��B�#�                                    Bx��v�  "          A"�\��{@��@�z�B+��B�33��{@�z�@�B4�\B�ff                                    Bx�݅j  "          A$  �z�@�(�@�\)B+B�#��z�@�33@�  B433B�z�                                    Bx�ݔ  T          A"{��
=@ȣ�@�ffBI\)B�LͿ�
=@�
=@�BR  B�W
                                    Bx�ݢ�  �          A(�Ϳ=p�@�=q@�\)B&�\B��Ϳ=p�@�@��B/33B�33                                    Bx�ݱ\  �          A+\)�(�@�@��
B'G�B��q�(�@��@��B/�HB�
=                                    Bx���  �          A(z῏\)@�ffA�\BT��BȊ=��\)@�(�A
{B]
=Bɞ�                                    Bx��Ψ  
�          A+���\)@�33A\)BcffB�p���\)@�Q�A�RBk��B�                                    Bx���N  
�          A+
=��ff@��HA��Bg\)B�.��ff@�Q�A�
Bo�\B�z�                                    Bx����  �          A)녿Ǯ@��A�Bc��B�Q�Ǯ@�\)A��Bk�RB�#�                                    Bx����  �          A#33�AG�@�\)A�RBa(�B��AG�@�A	G�Bg��B�z�                                    Bx��	@  �          A�H��33@���@�ffBJ�CJ=��33@p  @��HBO��CB�                                    Bx���  �          A33�|��@g�Ap�Be(�C	xR�|��@U�A\)BjffC�)                                    Bx��&�  
�          A (��R�\@��\A  BiQ�C �)�R�\@r�\A
=qBo\)C�R                                    Bx��52  
�          A����@�p�A�HBa�
B���@�z�Ap�Bh�
B�\)                                    Bx��C�  �          AQ��Q�@�  @���BS��B�B��Q�@��@�\)BZB�.                                    Bx��R~  "          A"ff��
=@���A\)Byz�B�=��
=@~�RA��B�(�B�R                                    Bx��a$  �          A"�\��@~�RA
=B��\B�LͿ�@k�A��B�\B�=q                                    Bx��o�  �          A"{�X��?z�HAB��\C#��X��?0��A=qB�p�C(xR                                    Bx��~p  T          A"�R��z�
=qA  Bm��C9����z�Q�A�Bl�C=�                                    Bx�ލ  
�          A"{���
?&ff@���BJ{C.33���
>���@�G�BJ��C0^�                                    Bx�ޛ�  �          A (��tz�
=A33B�u�C<���tz�^�RA�HB���C@�{                                    Bx�ުb  �          A\)�mp��B�\A33B�B�C?�\�mp����A�RB�p�CC�H                                    Bx�޹  "          A(��\(���33A��B�u�C9�f�\(��!G�Ap�B���C>W
                                    Bx��Ǯ  
�          A�\�6ff>��
A\)B�aHC-���6ff=L��A\)B���C3                                      Bx���T  
�          A
=��33?��A�B�(�Cz��33?��A��B��fCc�                                    Bx����  T          A�׿�ff@G�A\)B�� CǮ��ff?�\A(�B��)C�\                                    Bx���  
�          A=q��  ��p�A	G�B�CE���  ��(�A��B}��CH5�                                    Bx��F  �          A�����H���R@�(�B`=qCDY����H���H@��\B^\)CF�{                                    Bx���  �          A\)���׾u@�\BWC6s3���׾�G�@�\BW\)C8�{                                    Bx���  �          Ap���  @Q�@���B0�CL���  @G
=@ϮB3{C�)                                    Bx��.8  �          A=q���R@�(�@�ffB�C:����R@��@�=qBffC+�                                    Bx��<�  
�          A
=�\)���A�B�Q�C�E�\)�'
=A
�\B�aHC�k�                                    Bx��K�  �          Az�p����HA	B�� Cx��p���(Q�A��B�ǮCzn                                    Bx��Z*  T          A  ��(��(��A�
B�8RCu=q��(��6ffA�RB���Cv޸                                    Bx��h�  
�          A�\>#�
�EA��B�{C��H>#�
�R�\A\)B�G�C�j=                                    Bx��wv  
�          AQ�?�G��j�HA ��B���C�w
?�G��w
=@�
=B|�C�f                                    Bx�߆  
�          A=q?����xQ�@�G�BvC�0�?������@�{BqC��)                                    Bx�ߔ�  
Y          A��?^�R�vff@�=qB|C�e?^�R����@�\)Bw��C��                                    Bx�ߣh  
�          Az�����z�@�  Bq  C��
�������@���Bk�HC�H                                    Bx�߲  T          A	��    �w�@��B{{C��    ��G�@�{Bv
=C�H                                    Bx����  T          A=q���@��H@�G�B4�
B�ff���@�
=@�z�B9��B�
=                                    Bx���Z  "          A�\���@�G�@N{A�{Bה{���@�
=@W
=A�33B��f                                    Bx���   "          @�=q�4z�?�33@]p�B633C�3�4z�?�=q@`  B8��C��                                    Bx���  
�          Aff�����  @�Q�BCeB������33@��BG�Ce��                                    Bx���L  T          A
ff�����@�Q�B�HCeaH������@��Bp�Ce�                                    Bx��	�  "          A
�H�����G�@��
B#  C])�����z�@�G�B �C]��                                    Bx���  "          A������8��@ָRBG�\CR����AG�@��BE\)CS��                                    Bx��'>  T          Az���z���@أ�BJCJ  ��z����@�\)BI(�CK&f                                    Bx��5�  �          A(�����@�  Bm
=C::�����R@�Bl�C;�f                                    Bx��D�  �          AQ���
=?Y��@��HB[��C)����
=?:�H@�33B\��C+0�                                    Bx��S0  T          A���  ���
@���BOCA���  ���@��
BN�
CB�
                                    Bx��a�  S          A=q���W
=@��BE
=C<aH���p��@���BDz�C=n                                    Bx��p|  �          Ap���p���@�{B]��C:!H��p��#�
@�B]�\C;h�                                    Bx��"  	�          AQ���p��k�@�  BV(�C6����p�����@�  BV  C7��                                    Bx����  U          A���(�>���@��
BA��C0����(�>W
=@��
BA��C1�
                                    Bx���n  �          Az���=q@dz�@��HB��C�=��=q@`  @�z�B\)C                                      Bx��  
(          A{��(�@��R@�p�B��C\)��(�@���@�\)B
�RC�=                                    Bx�๺  �          A�
��{@�{@��A�C����{@�z�@��
A�33C                                      Bx���`  �          A33��G�@�z�?���@�ffB��
��G�@��
?�
=A��B��                                    Bx���  "          A\)�X��@��?�ff@�z�B�=�X��@�  ?���A  B瞸                                    Bx���  
�          A(��#�
A  �G����Bր �#�
A(��0����G�B�u�                                    Bx���R  "          A�H��  @�R>Ǯ@\)B�  ��  @�R>��@@��B�
=                                    Bx���  "          A(����H@���.{��=qB�3���H@�Q��(����{B�=                                    Bx���  T          A�����@����33�Q�B��H����@��R��G���B�ff                                    Bx�� D  !          A��QG�@C33��z��g��C���QG�@G����
�f\)C\)                                    Bx��.�  �          AQ��=p�@��\��z��9\)B����=p�@�z����H�7��B�z�                                    Bx��=�  �          A�׿�@�����33�M\)Bͮ��@�ff����K�\B�p�                                    Bx��L6  �          Ap���33@��\����4ffB��쾳33@��
����2�B�Ǯ                                    Bx��Z�  T          @��H@�@������H�
=B��@�@�=q�������B��                                    Bx��i�  �          A��@=q@������'��B�z�@=q@�����ff�&p�B��q                                    Bx��x(  �          A��@�p�@�z���ff�,ffBK@�p�@���p��+\)BL�\                                    Bx���  T          @���@�\?L�ͿaG���A�\)@�\?L�ͿaG����HA�=q                                    Bx��t  �          @��H��p����
@�{B�W
C6����p�����@�{B�Q�C7J=                                    Bx��  �          @ۅ���\@z=q?˅AW�
C�=���\@y��?���AZ{C�{                                    Bx���  T          A z�����@�@�p�BP��CW
����@@�BQ\)C�{                                    Bx���f  T          A���Q�@�33@EA�{Cn��Q�@��H@G
=A���C}q                                    Bx���  �          A���
=?��@�z�B+�RC)�f��
=?�ff@�z�B+�
C*�                                    Bx��޲  #          A����33?���@�  B0G�C)����33?�ff@�  B0\)C)Ǯ                                    Bx���X  �          A(����?�  @��BM�C�����?�  @��BM=qC��                                    Bx����  
(          A����@6ffAB�B�B�ff��@5A�B�W
B��                                    Bx��
�  	�          A(��)��@N�R@�Bx�HC^��)��@N�R@�Bx�CaH                                    Bx��J  "          A��AG�@Vff@��Bo�C\�AG�@Vff@��Bo{C
=                                    Bx��'�  �          A��<(��aG�A\)B���C8Q��<(��W
=A\)B���C8(�                                    Bx��6�  �          AG��J�H�#�
A�
B���C6���J�H�\)A�
B���C6��                                    Bx��E<  
�          A�
�333��@�ffB�#�C[#��333���@�ffB�G�CZ�                                    Bx��S�  �          A���%�33A�B�k�CZY��%��A�B���CZ\                                    Bx��b�  
�          A�R�*=q���HAp�B�k�CLn�*=q��
=A��B��{CK�3                                    Bx��q.  
�          A���R@.�RA�B�B�  ���R@1G�A�B�ffB�Q�                                    Bx���  	�          Ap���@EAp�B�33B����@HQ�AG�B�ǮB�                                    Bx��z  	`          A=q�(�@�z�@�ffBf{B����(�@�p�@�Be{B�{                                    Bx��   "          A�ÿ�=q@�=q@��BKQ�B�#׿�=q@��@���BJ{B��                                    Bx���  
�          A	���H@��@�
=BM�BӞ����H@�ff@�{BL33B�aH                                    Bx��l  
�          AG����
@��@߮Bfp�B�uÿ��
@��@޸RBe  B���                                    Bx���  
Z          A  �B�\@8��@ڏ\Bi��Cn�B�\@<(�@��Bh��C�                                    Bx��׸  �          A
=q��  @��H@��B-�\C����  @�z�@��
B,G�C��                                    Bx���^  
�          A\)���
@�Q�@z=qA�
=C����
@�G�@w�A�z�CT{                                    Bx���  T          A
ff��33@�\)�L�Ϳ�\)C	�R��33@�
=������HC	��                                    Bx���  T          A�
���@�\)@xQ�A��HC
G����@���@u�A��C
�                                    Bx��P  "          A����p�@I��@���B3\)C����p�@N{@��B2�C�                                    Bx�� �  �          AG���G�@
=q@�(�BM�
C�\��G�@�R@�33BL�
C�                                    Bx��/�  "          A����  >�(�@�z�BW��C/\��  ?�\@�(�BW\)C.(�                                    Bx��>B  
�          A{��{@�R@��B^z�CL���{@z�@�B]33Ch�                                    Bx��L�  �          A��33@B�\@���B=�RCk���33@G
=@�Q�B;��C�=                                    Bx��[�  "          Aff�*�H��G�@��HB��qCM=q�*�H��33@�33B�aHCK^�                                    Bx��j4  T          @�G���33�O\)@�ffB��=CRB���33�5@�
=B�L�CN�R                                    Bx��x�  �          A�H�p����A Q�B��3Cxp��p�����A ��B�B�Cwn                                    Bx�㇀  
�          Az����$z�A	B�  C��=����(�A
ffB��3C���                                    Bx��&  T          A\)>8Q���z�@�z�Bj��C��>8Q���Q�@��RBn�\C�)                                    Bx���  
Z          A�H���
�1�@�RB�{C�,ͽ��
�*=q@�  B���C�"�                                    Bx��r  
�          @��H�{�޸R@�B���CZ\�{��\)@޸RB���CX�                                    Bx���  
Y          @��fff�AG�@�(�BC�\C[���fff�:�H@�BF�C[{                                    Bx��о  �          @��R�Q����@�(�B;��Ct�f�Q����R@��RB?p�Cth�                                    Bx���d  
�          @�������R@333A��C�� ������@;�A��C��)                                    Bx���
  �          @��R��ff��
=@@��A�p�Ce����ff���@G
=A�(�Ce33                                    Bx����  
�          A��������@6ffA���CexR�������@=p�A��Ce+�                                    Bx��V  "          A�R�������R?���A��C\�)������?�ffA�C\��                                    Bx���  
(          Aff��\)���H?��A��CSٚ��\)���?�p�A#�
CS��                                    Bx��(�  
�          A����=q��{?��@�\)CW�f��=q���?�  A�CW                                    Bx��7H  
�          A33�ָR���ÿ.{��G�CV\�ָR�����z���=qCV#�                                    Bx��E�  
�          @�����z������.{��z�CUL���z����H�'����\CU�3                                    Bx��T�  "          @����G����\��(��J�HCV�{��G����
��{�>ffCV�{                                    Bx��c:  �          A ���Ӆ�~{�(��~=qCR�q�Ӆ�����ff�r�\CSQ�                                    Bx��q�  "          @�{�љ��p��g
=��ffCH�)�љ��#33�c33��=qCIL�                                    Bx�䀆  
�          A=q��Q�^�R������C;{��Q�}p����R� �HC<�                                    Bx��,  
Z          A��z�0���Tz���33C9&f��z�G��S33��  C9��                                    Bx���  "          @��R��>����<������C0���>��
�=p���33C1�=                                    Bx��x  T          A=q��G�?�{�Fff��G�C%�R��G�?�\�I����ffC&\)                                    Bx��  
�          @��H��R=�\)�8Q���z�C3h���R�#�
�8Q���z�C4\                                    Bx����  �          @���(�?�z��333��C*�q��(�?�=q�5���
C+c�                                    Bx���j  
�          @�ff���
@z��{��
=C"�
���
@   �!���
=C#.                                    Bx���  
�          @�Q���(�@��!�����C!L���(�@   �%��33C!�3                                    Bx����  �          A=q�ڏ\?��H�}p����HC%���ڏ\?˅��Q���=qC&��                                    Bx��\  
�          Ap���@!���Q����C����@�����H��Q�C�=                                    Bx��  
�          @�G�����@���$z����RC0�����@����-p���C�R                                    Bx��!�  "          @ٙ�����@�
=�ٙ��h��Cٚ����@�������~ffC5�                                    Bx��0N  �          @�=q�K�@�{� �����HB�Q��K�@��
�p���z�B��f                                    Bx��>�  �          @�=q���H@����G��ÅB�{���H@���S�
��{B�.                                    Bx��M�  �          @��\��p�@�  �=q����B����p�@���'
=���B���                                    Bx��\@  	�          A�?#�
>�  A�B�W
A�{?#�
?
=qA��B�8RB!�
                                    Bx��j�  
(          Ap�?(��?�RA�
B�=qB,�R?(��?n{A\)B�{B[p�                                    Bx��y�  �          A{<��
?=p�A  B�aHB�33<��
?��A�B�#�B�\)                                    Bx��2  
(          A���ff@EA�B���B�q��ff@Y��A��B��B���                                    Bx���  "          A
=�_\)@��@��B�
B��
�_\)@�Q�@�p�B33B�=q                                    Bx��~  
�          A\)?�G�?�A
{B�  A�z�?�G�?Y��A	B��B p�                                    Bx��$  �          Az�@�׾��RA
�\B�u�C�R@��<�A
�RB��q?Tz�                                    Bx����  �          A@'��\A��B���C��3@'��#�
A��B���C��\                                    Bx���p  �          A(�@�z��A�HBz
=C��f@�z�>��RA�RBy�R@���                                    Bx���  T          A@fff?�G�A�B���Ay��@fff?�\)A
�HB��A�Q�                                    Bx���  �          A=q@j=q@33A	��B�A�=q@j=q@=qA  B{  B��                                    Bx���b  
(          A�\@��H?��@љ�B-�Az�@��H?�\)@ϮB+��A4��                                    Bx��  �          A
=@��?У�@ə�B#�AP(�@��?�33@�
=B!z�Aqp�                                    Bx���  J          A\)@�
=?��@��B�A��@�
=?���@�33BG�A7�                                    Bx��)T  "          A!G�A  >W
=@�(�B	33?�
=A  >�@��B@K�                                    Bx��7�  �          Ap�@�33@�@�z�BN�\A�p�@�33@(Q�@���BJ\)A�33                                    Bx��F�  �          AG�>�=q@���@��HBaQ�B���>�=q@�z�@��HBV�RB�W
                                    Bx��UF  
�          A���33@���@�z�BQ��Bʅ��33@��
@�BG
=B�=q                                    Bx��c�  "          A�@�
=@�p�@��B��BQ{@�
=@�p�@�  A�(�BU=q                                    Bx��r�  �          A ��@�33@�\)@�p�A�\)B?(�@�33@�p�@uA��HBB\)                                    Bx��8  
�          A�H@�
=@��@G�AV�\B\  @�
=@�p�?��A1��B]ff                                    Bx���  
�          A�R@n{AQ�@\)A^{B��=@n{A�?�=qA4z�B�\                                    Bx�构  �          A=q@\)AQ�@��Af=qB�#�@\)A{?�
=A9p�B�u�                                    Bx��*  "          A=q@'�A\)@=qAa�B�=q@'�AG�?�Q�A4��B���                                    Bx���  "          A(�@�
A�
?�R@hQ�B�� @�
A(�=�?:�HB��\                                    Bx���v  "          A��?�\)Ap���\)���B�L�?�\)A(���{�4Q�B�.                                    Bx���  �          A�H� ��@��R@\)A��RB�q� ��@��H@
�HA�=qB��f                                    Bx����  
�          A\)��G�>.{@�Q�B

=C2�3��G�>�(�@�  B	�C0�
                                    Bx���h  
�          A
�\��  ?�@�  B1G�C�3��  @(�@�z�B,Q�C5�                                    Bx��  
�          A=q�?\)@��@��BCp�B��f�?\)@�33@��
B8z�B�p�                                    Bx���  �          A33�c33@���@��HB&
=B�{�c33@�\)@�  B  B�\)                                    Bx��"Z  
�          A�\�l(�@��
@��B0ffB���l(�@��R@�  B%��B�                                    Bx��1   
�          A{�ۅ@!G�@�z�B�\C���ۅ@5�@��B��C��                                    Bx��?�  �          A{��33@
=@��B'33C�)��33@-p�@��B"G�C��                                    Bx��NL  "          A���=q@Q�@�z�B*�C!ff��=q@�R@�Q�B&
=C��                                    Bx��\�  
(          A�����?���@�z�B633C#)���@��@ȣ�B1�HC�H                                    Bx��k�  "          A�����@  @˅B4�
C�����@(Q�@�
=B/��C�q                                    Bx��z>  �          A�R�Ϯ>#�
@�=qB1  C2���Ϯ?
=q@ə�B0G�C/=q                                    Bx���  T          A�\����@�\)?�A>�\C!H����@��H?��
A{C�{                                    Bx�痊  �          A���G�@�\)@i��A�\)C�{��G�@�ff@XQ�A��RC��                                    Bx��0  
�          A���@C33@��Bp�Cp���@X��@�ffB33C�                                    Bx���  
�          A�
��  @(��@���B&\)C���  @AG�@��HB Q�C:�                                    Bx���|  
Z          A��H��@�{?
=q@l(�B�\)�H��@��R=#�
>�\)B�G�                                    Bx���"  
�          A�H���A�����K�B����Aff�.{��\)B�                                    Bx����  
�          Ap���A(������7�
B���A�$z��s�B�=q                                    Bx���n  
�          A
=��\)AG���33�5BǨ���\)A�H�!G��qB���                                    Bx���  �          A�
�xQ�A�\�{�S\)B��ͿxQ�A�
�5��=qB�                                    Bx���  T          A33����@�G�@QG�A��\B��R����@�  @2�\A�Q�B�=q                                    Bx��`  
(          A�����@У�@�(�A�C �����@�G�@k�A��B���                                    Bx��*  "          A�\���@ۅ@��A��
B�����@��@�(�A�
=B�L�                                    Bx��8�  T          A  ���
@�p�@�=qA�B�k����
@�ff@w
=A�  B�                                    Bx��GR  
(          AQ����@��@eA��HB�����@�G�@C33A��RB�k�                                    Bx��U�  T          Az��r�\A{@��AnffB���r�\A��?�A2�RB��H                                    Bx��d�  "          A
=���\@�{@w
=A��B��
���\@�ff@S33A�=qB�8R                                    Bx��sD  
�          Ap��z=qA�H?�G�@���B�L��z=qA�>�Q�@�B�{                                    Bx���  
�          A���e�Ap����H�8��B���e�Az῔z��ٙ�B�33                                    Bx�萐  �          Ap��Tz�A=q������{B���Tz�AQ�����2�\B�ff                                    Bx��6  �          Ap��o\)A�þ�33��\B��)�o\)A(����
��Q�B�{                                    Bx���  T          Ap��<��A����R�  B�p��<��AG����M�B��                                    Bx�輂  �          A (��Q�A33�\)�f�RB�W
�Q�A��L����p�B��                                    Bx���(  T          A=q���A33�XQ����\B�����A
�\����\B��H                                    Bx����  "          A=q�L��A33��(�� 33B��q�L��@�����  �  B��f                                    Bx���t  T          Azἣ�
@��������B�=q���
@�ff��  � {B�=q                                    Bx���  
�          A����@��H���\�(�B��q���@����$=qB�L�                                    Bx���  T          A�����@�z���G��  B��׾���@�(��Ӆ�/G�B���                                    Bx��f  "          A�?���@����  �\33B��?���@�33�ff�m��B���                                    Bx��#  �          A�\��  @�(���G��$�B�G���  @�z�����7�B���                                    Bx��1�  T          Az��A   �����=qB��R��@���G��  B���                                    Bx��@X  �          A���=p�A  ��\)���
B�Q�=p�@��\��z��	B��)                                    Bx��N�  "          A  ��p�A
=�aG���Q�BøR��p�A	���  ��=qB�L�                                    Bx��]�  "          A
=�uA\)�������
B�ff�u@�����{�{B��                                    Bx��lJ  "          A�\�z�A(��(Q���33B���z�A (��Tz�����B���                                    Bx��z�  
�          A����(�A z�?L��@��B�=��(�A�=�G�?+�B�L�                                    Bx�鉖  
�          A33�A�A  �;���(�B�B��A�A\)�j=q��\)B܅                                    Bx��<  �          A���5A�׿�����G�B����5A
�\��{�8z�B�L�                                    Bx���  �          A����A33?\(�@�(�B�q���A�
=�?5B�                                     Bx�鵈  
�          Ap���@�\)���R���B�ff��@�=q�(��O�B��                                     Bx���.  �          A�
��{A Q��%��w�
B����{@�Q��Q����B�L�                                    Bx����  "          Aff����A (�����dz�B������@��������B��                                     Bx���z  �          A����33@�?�{@���C �3��33@�>�
=@\)C �3                                    Bx���   
�          AQ�����@�������G�B�  ����@��R�G��A��B�
=                                    Bx����  
�          A33���A�\��p��=G�B����@�{�.�R���HB�aH                                    Bx��l  
�          A
=����A��G����
B�=q����@�
=�\��B��                                    Bx��  T          A�R��=q@��Y����=qB��R��=q@�=q��=q�(�B�z�                                    Bx��*�  "          A33��
=A{�  �W\)B�=q��
=@�(��@  ���HB���                                    Bx��9^  �          A
=��=qA���'��{�B��
��=q@���W�����B�                                    Bx��H  T          A���G�Az��(��P��B�z���G�A z��>{����B���                                    Bx��V�  T          A����@�\)���H�	�B��
���@����{�S
=B�
=                                    Bx��eP  �          A33��{@������-��C ����{@�
=�!��r�RCxR                                    Bx��s�  
�          A�H��
=A����Q��8��B�\)��
=A��/\)���
B��3                                    Bx�ꂜ  �          A�H���A�Ϳ�(���  B�.���A=q���B{B�(�                                    Bx��B  T          A��j�HAG���(���B�.�j�HA
=q��_�B�\                                    Bx���  �          A���(�A  ��Q����B���(�A�����Y�B�\                                    Bx�ꮎ  "          AQ��|(�A
{����QG�B���|(�A��C�
���\B�z�                                    Bx��4  T          A���A33�Fff��
=B�����@��H�z=q��B���                                    Bx����  "          A�H���HA��У��Q�B�z����HA  �\)�ip�B�R                                    Bx��ڀ  "          A ����ff@�G������ۅC ����ff@��
���R�733C&f                                    Bx���&  
�          A����33@�z�?�
=A
ffCG���33@�Q�?Q�@��C��                                    Bx����  T          A��@�(�?�Az�C�q��@�  ?G�@�z�C5�                                    Bx��r  "          A�R���@Ϯ?��@��C	p����@��>�p�@{C	)                                    Bx��  �          A������@Ϯ>k�?�{C	J=����@�\)���0��C	Y�                                    Bx��#�  �          A Q��\@�ff��\�#�C E�\@�
=�%��o�
C!H                                    Bx��2d  
�          A!����RAp��Z=q��=qB�  ���R@���Q����B��3                                    Bx��A
  
�          A!���33@�33��Z{B�����33@��J�H���B�{                                    Bx��O�  
�          A!���HA����H�(�B�W
���H@���%�nffB��f                                    Bx��^V  �          A���  @�=q�����(�C����  @��Ϳ��H�733C0�                                    Bx��l�  
�          A�\���\@�(�����G�B�  ���\@�ff���=p�B�=q                                    Bx��{�  T          A
=��(�@��R�=p�����B�����(�@�\�u���p�B�.                                    Bx��H  
(          A   ���RA�H�1����
B�=���R@���k���\)B��)                                    Bx���  
�          A �����\A�R���R�7\)B�B����\A=q�:�H���B���                                    Bx�맔  
Z          A ����A	�������G�B��A=q�=q�_
=B��f                                    Bx��:  
�          A�
��G�@�\)�����=qB��H��G�@������=G�B��                                    Bx����  T          A���\)A�ÿ�����(�B�W
��\)A����R�RB���                                    Bx��ӆ  T          A (�����A	G���{�+�
B�B�����A���5���B��
                                    Bx���,  
�          A
=��=q@�\)�L�;�z�B�8R��=q@�p��xQ����B���                                    Bx����  "          A(�����@�G�?���A<��Cu�����@�\)?�@��C��                                    Bx���x  
�          A  ���@��?�33A4��C	�)���@��H?���@�
=C�{                                    Bx��  �          AQ���p�@��s33���C� ��p�@�=q��=q�,��C!H                                    Bx���  "          A���  @�33?��@ȣ�Cff��  @�p�>8Q�?���C�                                    Bx��+j  �          A{�ƸR@�ff?c�
@�=qC�=�ƸR@�Q�<��
=�C��                                    Bx��:  T          A��
=@��?�\)@�Q�C���
=@�  >�\)?ٙ�C�                                     Bx��H�  "          A����33@���?n{@��B�����33@��H<#�
=�\)B�=q                                    Bx��W\  
�          A=q���RAp�?�@E�B������RAp�����2�\B��{                                    Bx��f  �          A33��z�@�(�?#�
@r�\B�Q���z�@��;�����\B�.                                    Bx��t�  "          A(����A>�=q?�{B����Ap��B�\���RB�Ǯ                                    Bx��N  �          A�����A��?&ff@tz�B�Q�����A�þ\���B�8R                                    Bx���  "          A�
���A(�>��R?���B� ���A�
�8Q���Q�B��                                    Bx�젚  �          A����HA ��?W
=@�\)B�\���HA���.{���\B���                                    Bx��@  �          A\)����A�;���fffB�p�����A\)��Q���=qB���                                    Bx���  �          A���G�Az�8Q���  B�\��G�A녿���*ffB���                                    Bx��̌  T          A�\����A(�>�p�@p�B������A��5���B�\                                    Bx���2  T          A����\A?���Ap�B�ff���\A�
>�@8Q�B��                                    Bx����  
�          Ap���=qA�
?˅A�B�.��=qA{?�\@@  B�W
                                    Bx���~  "          A  ��p�@�@!�Ap��B�=q��p�@�Q�?��AG�B�W
                                    Bx��$  �          A�H��(�@�\)@P  A���B��
��(�@��H@  AW�B�\)                                    Bx���  
Z          Az����@�
=@>�RA�ffB�.���@�@   AC\)B�Ǯ                                    Bx��$p  �          A=q���
@�ff@<(�A�G�B�8R���
@���@   AQ�B���                                    Bx��3  "          A	����@ҏ\@C�
A�(�B������@�@
=qApz�B�\)                                    Bx��A�  
�          A����=q@�33�AG���ffCB���=q@����j�H��(�C޸                                    Bx��Pb  
�          Ap����
@@������*�
C�����
@  ���H�8�
CL�                                    Bx��_  �          A�����@�Q�������C	c����@W�����(�\CL�                                    Bx��m�  �          A
=���\@���33��\C�
���\@��\��
=��HCE                                    Bx��|T  
�          AQ���=q@�(�?���@�ffC���=q@�
=>B�\?��C�R                                    Bx���  �          AG���p�@��@ffA\��C=q��p�@�z�?��\A�C#�                                    Bx�홠  
�          A{��G�@��@�=qA��HC	z���G�@�=q@S33A��C޸                                    Bx���F  
�          A{��  @��H@�ffA��C����  @��@j�HA�C�3                                    Bx����  T          Az���{@�
=@��\A�  C#���{@���@a�A���Ch�                                    Bx��Œ  T          Ap����H@��@qG�A�C���H@�ff@9��A�{C�H                                    Bx���8  T          AG���Q�@��H?�G�@�p�B�L���Q�@�{>aG�?�{B��=                                    Bx����  �          A�R���
A{?J=q@��\B�.���
A�\��Q��{B�                                      Bx���  �          A33��@�p�@z�HAģ�C}q��@��@>{A�
=Ch�                                    Bx�� *  �          A(��У�@�  @>�RA�Q�C8R�У�@Ӆ@�
AH��C�)                                    Bx���  T          AQ��׮@�Q�@9��A��C���׮@��
@AR=qC
�3                                    Bx��v  �          A���{@��
�k�����C����{@�׿�  ��33Cff                                    Bx��,  	�          A33����@�?�
=A�B������@�(�>�p�@B�#�                                    Bx��:�  �          A  ��z�@�R@Y��A�  B�����z�@�(�@z�Ab=qB��                                    Bx��Ih  �          AQ�����@�
=?aG�@�33C�����@��þ���c�
C�\                                    Bx��X  �          Az��׮@�z�n{���HCh��׮@�����7�CQ�                                    Bx��f�  T          A����A �Ϳ�p����B����@�������i��B���                                    Bx��uZ  T          A������A Q쿷
=�	��B������@�
=�&ff�|z�B�{                                    Bx��   �          Az���Q�@�\)���H��
=C����Q�@�\)���O33C�q                                    Bx�  T          A����@�(��  �\(�C33���@�\)�Fff��33C.                                    Bx��L  
Z          A���z�@��H�7
=���RC����z�@��H�mp���=qC33                                    Bx���  T          A���ff@�z�����>�\Cc���ff@����8����ffC
{                                    Bx�  T          A�
��{@��?�(�A�
C����{@陚>�p�@  C�                                    Bx���>  �          A�����@�@Q�AN�HC ���@�\)?�  @��B��3                                    Bx����  �          A���
=@�׿0����ffB��=��
=@�=q��=q�2�RC                                       Bx���  T          A�
��z�@���?:�H@���C�3��z�@�녾�Q��  C�
                                    Bx���0  T          A�
���@��@EA���C	@ ���@�
=@�AO33Cff                                    Bx���  �          A�H��{@�33@8Q�A�Q�C	� ��{@�\)?�Q�AC
=C��                                    Bx��|  "          A
=���@�33@�(�B�\C(����@�p�@��A��C�R                                    Bx��%"  
�          A{��G�@�z�@�z�A�33Cp���G�@�
=@Mp�A��\C��                                    Bx��3�  �          A�R�Y��@�(�@���B==qB��{�Y��@���@�
=B��B���                                    Bx��Bn  �          A���(�@���@ȣ�B)
=B����(�@�ff@�=qB33B�G�                                    Bx��Q  
�          A��33@�  @�\)B�C�)��33@ə�@�G�A��C                                    Bx��_�  "          Az����R@�{@�p�A�G�C����R@��@XQ�A�(�B��
                                    Bx��n`  
Z          A�
��p�@��
@�G�B�RCٚ��p�@�\)@�z�B33Cs3                                    Bx��}  �          A���s�
@�  @�(�B5�RB���s�
@�ff@��BffB�=q                                    Bx�  �          A�\��@�=q@�=qB^\)B��H��@�{@ϮB<  B��                                    Bx��R  �          Az��J=q@�Q�@�BJ  B�\�J=q@�=q@�=qB*
=B��                                    Bx���  "          A�R�!�@�Q�@�
=BaB�k��!�@��@�p�B@��B�{                                    Bx�﷞  T          Ap���ff@��@���A���C�
��ff@�
=@aG�A���C ��                                    Bx���D  "          A�����@��@�z�B
�CQ�����@�Q�@�
=A�{Cc�                                    Bx����  
�          A���33@�(�@�G�Bz�CJ=��33@�Q�@��B(�C�                                    Bx���  "          A�����@���@�(�B*ffC� ����@�33@���B��Cs3                                    Bx���6  �          A����(�@�=q@��B(Cٚ��(�@�Q�@�B  C ��                                    Bx�� �  "          A�R����@�G�@�G�B4�\Cs3����@���@�{BffB���                                    Bx���  T          A
=�r�\@��@�p�B\)B��f�r�\@�@��A���B�33                                    Bx��(  "          A��@�\)@~�RA�ffB��H��@��@8Q�A��B�\                                    Bx��,�  �          A
=�\(�@�@��BGffB�33�\(�@�  @�Q�B �B                                     Bx��;t  �          A33�&ff@��@��
B\)BݸR�&ff@�  @Y��A�p�B�33                                    Bx��J  �          A��<(�@��@�G�B"��B�q�<(�@�ff@�p�A��B���                                    Bx��X�  �          A(��L(�@�p�@\)Aי�B����L(�@��@-p�A�G�B�                                    Bx��gf  "          A�R�K�@�p�@��B{B��K�@�ff@|(�AԸRB��                                    Bx��v  
�          A��y��@���@_\)A�=qB�  �y��@���@  AqG�B��                                    Bx����  
�          A{��z�@��
@�B��B����z�A@e�A��B�p�                                    Bx��X  T          A
=��
=@��H@��B��Bх��
=A�@p��A�{B���                                    Bx���  "          A\)�5@�{@�
=B3�B��׿5@�R@�  B
=B��                                    Bx��  �          A\)��  @�=q@ƸRB*�\Bͳ3��  @��@��RB33B�p�                                    Bx��J  "          A���p�@Ӆ@�
=B+(�B����p�@�33@�
=B\)B�ff                                    Bx����  "          A��\)@߮@�B ��B�#׿\)@�p�@�33A�RB�\                                    Bx��ܖ  
�          A33�u@�=q@�\)B(�B��u@�
=@���A���B��                                     Bx���<  
�          A33����@�
=@�ffB5p�B̀ ����@��@��B\)B�\                                    Bx����  T          A�\�vff@�
=@)��A��HB�
=�vffA�?��R@�z�B�
=                                    Bx���  
�          Ap���=q@��
?���A<z�B����=q@�\?�@VffB�W
                                    Bx��.  
�          A=q�z�H@�\)?�A:�RB�L��z�HA�H>\@�B�.                                    Bx��%�  
�          A�R�Z�HA ��?\(�@��
B����Z�HAG�����z�HB��
                                    Bx��4z  
�          A������@�=q?�33A8��B������@�\)>W
=?���B�ff                                    Bx��C   T          A
=�33A
{�0�����RBή�33A��  �iBπ                                     Bx��Q�  
�          @�
=���@��@O\)A�{Bי����@��@
�HA��B��
                                    Bx��`l  �          @�\)�\)?��H@�\B��B�#׾\)@C�
@�z�B��B��                                    Bx��o  T          @��>���@�@���B+z�B���>���@�Q�@~{BQ�B��f                                    Bx��}�  !          A z�?���@���(��g�B��R?���@�\)��33�aG�B�.                                    Bx��^  
�          Aff@W�@������
���Bu(�@W�@�{��p��*(�Bd�                                    Bx��  "          A�@W
=@�p����G�Bq�@W
=@�  ��ff�(��B`Q�                                    Bx��  �          Ap�@>�R@�p���{���By=q@>�R@�������;�Be��                                    Bx��P  �          A (�@C33@�
=��
=�,��Bj{@C33@w��ə��O��BO{                                    Bx����  "          AG�@J�H@����Q��+��Bf=q@J�H@w����H�N�
BJ�                                    Bx��՜  �          @�\)@P��@x����p��JQ�BH33@P��@*=q��G��h�B                                      Bx���B  �          AG�@U�@k���ff�U�
B?�@U�@ff����r�HB{                                    Bx����  �          A\)@0  @^�R���
�d��BN�@0  @����B�                                    Bx���  
�          A�@�@P������s�RB\33@�?�������
=B�H                                    Bx��4  
�          A�?��@���׮�`�B���?��@@  ��ffffB�L�                                    Bx���  �          @�\)>���@��R�ə��Z�B�u�>���@J�H����.B�                                    Bx��-�  T          A�>��R@�����z��8�HB�� >��R@�
=��=q�d��B��                                    Bx��<&  
�          Aff>�z�@�Q����R�2�B�(�>�z�@�����
=�_33B���                                    Bx��J�  T          A�?=p�@����Ϯ�U33B��?=p�@\(���G�z�B��)                                    Bx��Yr  
�          A�\?5@�=q��(��f=qB�ff?5@9�����H  B���                                    Bx��h  �          A=q?^�R@�33�߮�lG�B�?^�R@*=q������B���                                    Bx��v�  "          A  ?Ǯ@Z�H���H�{��B���?Ǯ?����{BK�                                    Bx��d  "          A��?���@j�H��Q��t��B�  ?���@����H��BW33                                    Bx��
  	�          A�
?�(�@n�R��(��pQ�B�aH?�(�@����
=��BR�\                                    Bx��  �          A�
?�@Tz��陚�z�BtQ�?�?������\)B3(�                                    Bx��V  �          Aff?��
@E�����B��3?��
?�ff��z�Q�BI�                                    Bx���  
�          A(�?�G�@\����\)B�B�?�G�?�z����R��Bx=q                                    Bx��΢  �          A  ��@�  ����k�B����@0����Q��
B�Ǯ                                    Bx���H  
(          A�\��{@�33�ȣ��Iz�B��H��{@`  ���sffB���                                    Bx����  
�          A�R��z�@��\���H�A�B�LͿ�z�@p����\)�k\)B��                                    Bx����  �          A{��
@�
=�����5z�B�����
@|���ָR�^�B��                                    Bx��	:  "          A
=�&ff@�=q���
�7G�B�G��&ff@r�\�أ��_=qB��f                                    Bx���  �          A
=��H@�����
=�1��B�  ��H@�G���p��[33B�Ǯ                                    Bx��&�  "          A(����H@�����G��<{Bܮ���H@|����\)�g  B�R                                    Bx��5,  "          A��Fff@��H�����G�B�{�Fff@����ə��F  B�\)                                    Bx��C�  
Z          A{�E�@����
=�#��B�\�E�@�Q���
=�K��B���                                    Bx��Rx  T          A�����R@��
��Q���
C ����R@��H��Q��)ffC&f                                    Bx��a  "          Ap��Fff@�(���(��(�B랸�Fff@�Q���\)�9��B�8R                                    Bx��o�  
Z          A���G�@��\�����{C �
��G�@�\)��G��/Q�Cc�                                    Bx��~j  T          Aff��{@\�������B�.��{@��
��
=�  CW
                                    Bx��  "          A�����@�Q��`  �Ə\B�B����@������R�	��C #�                                    Bx��  �          A���@�(�������z�C
���@�G��(��u��C�=                                    Bx��\  �          Aff����@���n{��Q�B�G�����@�����H�33C�)                                    Bx��  T          A33�{�@ƸR�����B����{�@��R�����B�                                      Bx��Ǩ  
�          Aff���@�G����
�=qC�����@w�����;p�C	)                                    Bx���N  T          A{���
@�G����R� \)B�����
@�\)��=q�&ffC
                                    Bx����  
�          A���c33@�G������B�L��c33@���Å�=z�C.                                    Bx���  "          A��   @����  �5�RBۣ��   @�p������b��B�L�                                    Bx��@  	�          A�Ϳ�p�@�{��{�K��Bڔ{��p�@^{���H�x��B���                                    Bx���  
Z          A��=q@�Q���G��N\)B��쿪=q@`����ff�|��Bݙ�                                    Bx���  T          A���z�@�Q���ff�Jp�B��׿z�@qG���p��{
=BŊ=                                    Bx��.2  
Z          A�����@�33��
=�E  B��{����@y����
=�u��B��q                                    Bx��<�  T          Aff�333@��\�Ϯ�T33BĔ{�333@U���
\)B��
                                    Bx��K~  T          A{���H@�G��У��V33B����H@R�\������B��                                    Bx��Z$  
Z          A{��@����ff�^\)B�=q��@@�������B�ff                                    Bx��h�  �          Aff>W
=@�����{�k  B�
=>W
=@!G�����L�B�k�                                    Bx��wp  �          A�R<#�
@���ff�\33B���<#�
@G���B��R                                    Bx��  �          A�\�aG�@���Ӆ�Y\)B�ff�aG�@L����\)��B�                                      Bx����  T          A�R��@����z��gQ�B�(���@-p����z�B�8R                                    Bx���b  T          A�H���
@�����H�X��B�p����
@L����R�{B���                                    Bx���  "          A=q�z�H@�{�����M{Bʀ �z�H@\����R�~{B�Ǯ                                    Bx����  T          A�\����@��R��p��Gz�B�
=����@l(�����wz�B��                                    Bx���T  �          A��z�@����z��R
=B�  ��z�@O\)����aHB�#�                                    Bx����  T          A=q��  @�ff�׮�_�HB�33��  @'���  p�B�u�                                    Bx���  �          A����(�@�  ��{�`Q�B�#׿�(�@+���
=�B�{                                    Bx���F  
�          A녿�33@���ڏ\�e\)B�=q��33@�����Q�B��                                    Bx��	�  T          Ap���ff@�{��G��dz�B�ff��ff@%��\B�L�                                    Bx���  T          A �׿�  @z�H����i\)B��
��  @�
��Q��B��f                                    Bx��'8  
�          A Q���
@qG���(��m��B�(����
@����G��B���                                    Bx��5�  �          AG����R@`����  �lp�B�����R?�����C�                                    Bx��D�  "          A ���XQ�@.�R��\)�d�RC!H�XQ�?�33��p��~�C!0�                                    Bx��S*  
�          @�{��{?�z���{�N��C#O\��{���
��33�V(�C433                                    Bx��a�  
�          @�z���\)?�����Q��F�\C':���\)��  �Å�K{C6޸                                    Bx��pv  �          @�(����R?�=q��33�Z��Cs3���R=�G���G��d�C2��                                    Bx��  "          @�����\)?#�
�θR�a�
C+Y���\)�Tz���{�`�RC?�                                    Bx����  �          @�G��XQ�?��
��  �r  C@ �XQ�>W
=��\)8RC0�                                    Bx���h  �          @�=q�J�H?���{�z��C8R�J�H=u��z�
=C3                                      Bx���  T          @���0  ?�����RCz��0  >k���33�)C/L�                                    Bx����  �          @�=q�)��?�Q���\L�CǮ�)��>�=q���H#�C.
                                    Bx���Z  �          @��\�{@(���\  C
s3�{?���z�L�C(8R                                    Bx���   �          @��
�%�@33������C
W
�%�?�R����� C&xR                                    Bx���  T          @�{��@o\)����e=qB�ff��@������C��                                    Bx���L  T          @��R��p�@�{��
=�Y��B�{��p�@%��Q��B���                                    Bx���  �          @�ff���R@����ʏ\�S�B؀ ���R@<����ff��B�                                    Bx���  �          @����G�@���=q�U��BٸR��G�@6ff��p��{B��                                    Bx�� >  "          A���G�@��H���H�N�HB�.��G�@O\)��G���B�z�                                    Bx��.�  
�          A(����@�(���z��@B��Ϳ��@s�
��
=�t��B�Ǯ                                    Bx��=�  S          @�\)�Y��@��\�����>33B��Y��@s�
��\)�s
=B�                                     Bx��L0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��Z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��i|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��x"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��ެ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��'�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��E6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��S�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��q(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��ײ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��><              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��L�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��j.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��x�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��и              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��7B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��c4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��ɾ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��0H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��\:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��)N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��U@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��"T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��NF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��z8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��*               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��GL              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��d�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��s>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��#              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��@R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��lD              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  �          @�ff�c33@`  ���
�F�Ck��c33?��H�Ӆ�l�CT{                                    Bx���(  T          @��˅@����U��B���˅@�33��Q��7  B�\                                    Bx����  �          @�\?z�H@�=q?xQ�A ��B���?z�H@�=q�z�H�{B���                                    Bx���t  �          @�R?
=q@�  >k�?�z�B�.?
=q@�G���  �fffB��f                                    Bx���  �          @�ff��Q�@�=q������
B�ff��Q�@�������\)B���                                    Bx����  �          @�33��G�@�R�����$(�B����G�@أ��Y���ԸRB��                                     Bx��f  �          @�(��#�
@�\�Tz���  B�uþ#�
@�G��;�����B���                                    Bx��  �          @�=q<#�
@�p�����$Q�B��H<#�
@׮�XQ���33B��)                                    Bx��*�  �          @��þ���@�Q��\)�g\)B�=q����@�{�vff����B��f                                    Bx��9X  �          @�\��=q@���\�Xz�B�.��=q@���qG���G�B��                                    Bx��G�  �          @�녾�Q�@��Ϳ�33�+�B��׾�Q�@�ff�\(����HB�8R                                    Bx��V�  �          @�=q=L��@��H�����_33B�33=L��@����tz�����B��                                    Bx��eJ  �          @�z�>8Q�@�=q�33�y�B�=q>8Q�@�{����� �B��H                                    Bx��s�  �          @�(����
@��
��Q��l(�B�G����
@У��|����p�B�Q�                                    Bx����  �          @���z�@�z��$z���z�B��3��z�@��
��\)��
B�z�                                    Bx���<  �          @���>�33@��%���RB�ff>�33@����������B�u�                                    Bx����  T          @��?(�@����G���B�ff?(�@��H��\)�\)B�                                      Bx����  T          @�?G�@�����
�9p�B���?G�@��c�
��\)B���                                    Bx���.  �          @�G��#�
@�{������B�8R�#�
@�����z���RB�=q                                    Bx����  �          @��þ���@����33��z�B��;���@ƸR��\)��B��                                    Bx���z  �          @�\)�#�
@أ��Dz���{B��
�#�
@�(�����"p�B�
=                                    Bx���   T          @�p��#�
@�G��Tz���ffB�33�#�
@��\��G��,��B��R                                    Bx����  �          @��Ϳk�@�ff�dz����
B�LͿk�@�ff��G��?�B�G�                                    Bx� l  �          @���&ff@��\��G���B�B��&ff@K�����S=qC\)                                    Bx�   
�          @�Q��#�
@�
=���R�
=B�=�#�
@dz���ff�Lp�B�L�                                    Bx� #�  T          @߮�'�@��\��=q��
B���'�@Z=q��Q��PQ�B��                                    Bx� 2^  "          @ڏ\�
=@�������B�(��
=@e���Q��K=qB��3                                    Bx� A  
�          @���  @�(��`  ��=qB�uÿ�  @�(���33�3�B��                                    Bx� O�  T          @�p��G�@���w
=��  B�z��G�@�ff����7
=B�W
                                    Bx� ^P  
�          @�
=��{@����j=q��\Bӽq��{@�\)����0{B��                                    Bx� l�  �          @�����H@ۅ�P����p�B��ÿ��H@�����=q�#�B��H                                    Bx� {�  �          @�
=��Q�@�33�G����HBϮ��Q�@����{B�{                                    Bx� �B  �          @�p���33@��
�5���RB��ÿ�33@�G���33�  B��                                    Bx� ��  �          @��H��
=@�G���p��@��B���
=@��H�Vff���B�ff                                    Bx� ��  �          @�녿�{@߮�
=��B��ÿ�{@�����  ��HB��                                    Bx� �4  �          @�׿�Q�@�
=�=q��B�aH��Q�@�  ��G��=qB�
=                                    Bx� ��  �          @��ÿ�ff@��ÿ�=q�H��B�𤿆ff@�G��`����RB��f                                    Bx� Ӏ  �          @�R�G�@����
=�pQ�B�z�G�@�=q�x����33B��                                    Bx� �&  �          @�G���R@�Q��z��j�\B��3��R@�p��x������B���                                    Bx� ��  �          @�녾Ǯ@��
��=q�AG�B�  �Ǯ@��
�g
=��Q�B��                                    Bx� �r  �          @񙚾k�@�33�ٙ��O�
B����k�@�=q�mp���B�{                                    Bx�  T          @�녿!G�@�=q��  �W
=B��)�!G�@У��p�����HB�\                                    Bx��  �          @�\�(�@�녿��`��B�� �(�@Ϯ�u���
B��3                                    Bx�+d  �          @��H�W
=@��H��Q��NffB�\�W
=@���mp���(�B�                                    Bx�:
  �          @�=q���H@�  ���]�B��f���H@�{�r�\��z�B�8R                                    Bx�H�  �          @��H���H@�녿У��F�\Bƽq���H@љ��h����G�B��)                                    Bx�WV  �          @�(��s33@�����X��B��s33@љ��s33����Bď\                                    Bx�e�  �          @��\(�@�\���\��B�p��\(�@�Q��s�
��33B��                                    Bx�t�  �          @�녾�
=@�p���=q�"�RB����
=@�\)�XQ�����B�.                                    Bx��H  �          @��þ���@�33��G��8��B�33����@�(��a���(�B�                                    Bx���  �          @�>8Q�@�\)�Tz��˅B�8R>8Q�@�{�:�H����B�                                    Bx���  �          @�\)��@�z�W
=�θRB�
=��@�33�9�����B��                                    Bx��:  T          @��;k�@�ff�Ǯ�C�B��3�k�@θR�b�\��B��                                    Bx���  �          @�{���H@��Ϳ����s\)B��)���H@ə��y�����B��                                    Bx�̆  �          @�  ��(�@��ÿ�
=�P  B����(�@�Q��k���p�B���                                    Bx��,  �          @�
=���@�(��{��(�B�aH���@ƸR������B��q                                    Bx���  �          @�>���@�=q���]��B�G�>���@�Q��s�
��Q�B��                                    Bx��x  �          @�
=?�33@��R�   �fffB�Q�?�33@�  �*=q��z�B���                                    Bx�  �          @��R?�p�@��Ϳ��
�  B��?�p�@�\)�X����(�B�#�                                    Bx��  �          @��H?��H@�G����
�4Q�B�\?��H@ٙ��fff��33B��q                                    Bx�$j  �          @��H?���@�R���V�\B��?���@�z��vff��=qB�L�                                    Bx�3  �          @���?�
=@�33��\�O�B�Q�?�
=@�G��vff�陚B�B�                                    Bx�A�  �          @�=q?+�@�Q��z��uG�B���?+�@��
�������B�k�                                    Bx�P\  �          @�z�?8Q�@��
=q�~�RB�#�?8Q�@�(����R�{B���                                    Bx�_  �          @�p�?333@�(����H�f{B��{?333@�Q���G����B�B�                                    Bx�m�  
�          @�?���@�Q������33B��?���@ҏ\��\)��B��{                                    Bx�|N  �          @��H?B�\@���,(����RB�.?B�\@�  ������B�33                                    Bx���  �          @���?O\)@�33�.�R��=qB�ff?O\)@�����ff�\)B�B�                                    Bx���  �          @�?��
@��U����B��=?��
@��
��ff�$p�B�#�                                    Bx��@  �          @���?��@޸R�^�R��\)B�(�?��@�{����(�HB�p�                                    Bx���  �          @���k�@љ���Q����
B�{�k�@�����{�=B�\)                                    Bx�Ō  T          @�׿(�@����x������B��H�(�@�G���G��>
=B���                                    Bx��2  �          @��
>��@أ��[�����B�k�>��@�����ff�,��B��)                                    Bx���  �          @�G�>.{@�
=�q���\B�
=>.{@�z���ff�:��B�G�                                    Bx��~  �          @�p���ff@Ӆ�~{���B��;�ff@�\)��p��<�B��)                                    Bx� $  �          @�������@Ӆ���\����B�  ����@�ff�����?B��                                    Bx��  �          @�\)��\@�{�s�
��
=B��Ϳ�\@��H��G��7B�                                      Bx�p  �          @��׾�p�@�33�HQ���  B�녾�p�@���  � \)B��                                    Bx�,  �          @��׾\@��1G���p�B��\@�p���ff�
=B�\                                    Bx�:�  �          @�G����@����Q���Q�B�G����@�����ffB�aH                                    Bx�Ib  �          @�\)>�@�  ��
=�,��B��>�@�G��_\)����B�33                                    Bx�X  T          @��h��@�z�������\)B�#׿h��@�  �����@(�BȊ=                                    Bx�f�  �          @�Q쿫�@��R�����{B�W
���@�(���Q��WG�B�
=                                    Bx�uT  �          @��׿�(�@�(���Q��  BɸR��(�@���(��D��B��f                                    Bx���  �          @�
=����@ָR�g���=qBƨ�����@�����0Q�B���                                    Bx���  �          @�=q�}p�@�  �U���B�\�}p�@�G�����%Q�B�aH                                    Bx��F  �          @����
=@���s�
��(�B�W
��
=@�=q��33�2�B��f                                    Bx���  �          A �׿�
=@�33�e��{B�녿�
=@����{�*�B�                                    Bx���  �          Ap�����@���q����B�8R����@���33�/Q�B�{                                    Bx��8  �          Ap���
=@ᙚ�k�����B�녿�
=@����Q��+��B�                                      Bx���  �          A Q��\)@�=q�xQ���BΨ���\)@��R��(��2��B���                                    Bx��  T          @�����\@�33��ff�
  Bʮ���\@�33�����I\)B�k�                                    Bx��*  �          @�
=����@�=q�����B�G�����@�G����
�R
=Bр                                     Bx��  �          @��ÿ��\@��������+�B��
���\@vff��(��jQ�B؞�                                    Bx�v  �          @��ÿ���@��R��33�%
=B��f����@���Ϯ�dQ�B�u�                                    Bx�%  �          @�(���33@��
���H���HBǨ���33@��R�����={B���                                    Bx�3�  �          @�ff�!G�@������B���!G�@�=q�\�G�B�.                                    Bx�Bh  �          @��R�z�H@�(������Q�B��H�z�H@����R�B(�BɅ                                    Bx�Q  �          @�\)���@�{��{��  Bƀ ���@�Q���z��>��B�p�                                    Bx�_�  �          A   ��(�@ָR�\)��p�BиR��(�@��\��ff�5�B׳3                                    Bx�nZ  �          @����R@�
=����G�B��ÿ�R@�{��\)�M�B�W
                                    Bx�}   �          @�p���\)@�  ���
��B�.��\)@�������X  B��                                    Bx���  T          @�{�W
=@����33� G�B��H�W
=@����ҏ\�`�RB�W
                                    Bx��L  T          A\)��z�@˅����(�B�.��z�@���\)�\�\B�                                    Bx���  �          A�H��Q�@ڏ\���H� 
=B�Ǯ��Q�@������?G�B�
=                                    Bx���  �          @����H@�  �q���RB�=q���H@�ff���4��B�L�                                    Bx��>  �          @У��:�H@�G���(��u��B�\�:�H@��H�K���
=B��                                    Bx���  �          @����@���O\)���B�Ǯ���@�  ��33�.�RB�
=                                    Bx��  �          @�Q���R@�ff����  B�
=��R@��\�u�
�\B�G�                                    Bx��0  �          @����=q@����G���  Bߞ��=q@�p��vff���B�\                                    Bx� �  �          @��
�H@���AG���\)B�k��
�H@�����\� \)B�                                    Bx�|  �          @�p���(�@�������\)B����(�@�=q����?Q�B�\                                    Bx�"  �          @��\��\)@��
����
B�.��\)@��H���P�
B�#�                                    Bx�,�  �          @��
�xQ�@�p���ff��RB�z�xQ�@�G���33�Oz�B͏\                                    Bx�;n  �          @�����@�\)�����'G�B�z῅�@n{�����e�B�B�                                    Bx�J  T          @��þ�(�@�
=���H���B��{��(�@�(���  �Mp�B�33                                    Bx�X�  T          @�  �u@�33��{��RB���u@��H���
�Zp�B�L�                                    Bx�g`  �          @��H?�33@�������B�\B��q?�33@-p����H�}��Bz�                                    Bx�v  �          @Ӆ?�z�@X����G��d�B�8R?�z�?޸R��Q�\Ba=q                                    Bx���  �          @���?���@J=q��ff�l�B�Ǯ?���?�p���33k�BA�                                    Bx��R  �          @�  ?���@AG���33�q�RB��
?���?�ff��ff8RB-G�                                    Bx���  �          @��H?��R@�
�ə���Bwz�?��R?   ��{u�A���                                    Bx���  �          @Ϯ?+�@#33��p�\B��=?+�?h������33BV{                                    Bx��D  �          @�z�p��@dz���ff�Z��B�z�p��@ ����\)8RB�                                    Bx���  �          @�p��aG�@g
=����Z�B�uÿaG�@�\����z�B�3                                    Bx�ܐ  �          @�{�0��@:=q��ff�yG�B��H�0��?�  ����ffB�W
                                    Bx��6  �          @�Q�:�H@q���\)�V�B��:�H@p�����fB؞�                                    Bx���  
�          @�
=��(�@qG�����P��B׽q��(�@\)��{z�B��                                    Bx��  �          @�ff���@}p����R�I��B��)���@p���33ǮB�k�                                    Bx�(  �          @У׾�G�@[������g��B ��G�?��Ǯ��B�#�                                    Bx�%�  �          @θR���
@i�����R�W��B�p����
@ff��  � B�#�                                    Bx�4t  �          @�p���ff@p�����I��B����ff@�\�����qB�G�                                    Bx�C  �          @θR�xQ�@�(�����7{B�\�xQ�@>{���
�s�RB�G�                                    Bx�Q�  �          @�\)����@����ff�:(�B�uÿ���@8Q���{�u�HB���                                    Bx�`f  �          @�Q�aG�@��
���H�=�B�
=�aG�@.{�����z�Bר�                                    Bx�o  �          @�ff��z�@��H�\)� p�B���z�@Fff�����X
=B�=q                                    Bx�}�  �          @���ff@�=q�����2=qBۏ\��ff@<(���Q��k��B��                                    Bx��X  �          @�p�����@��\��=q�3�B�(�����@1G���Q��m{B�W
                                    Bx���  �          @�G���@U����9�B�W
��@��p��j{C=q                                    Bx���  �          @�Q���H@S33��G��2�\B�aH��H@ff�����`��C
=                                    Bx��J  T          @�z��(�@Y���z�H�0p�B���(�@�R��{�affC��                                    Bx���  �          @��\�'�@]p��_\)�Q�B�aH�'�@����G��Iz�C	�
                                    Bx�Ֆ  �          @�=q�8Q�@S�
�X���z�C  �8Q�@�\�����A�C��                                    Bx��<  �          @�Q��;�@8Q��j�H�'(�C}q�;�?�������M�RC��                                    Bx���  �          @���(�@C�
�����;B�W
�(�?�\)��ff�iC�                                    Bx��  �          @��
��G�@Mp�����K\)B�ff��G�?�
=����33B���                                    Bx�.  �          @�(���33@G������FG�B�׿�33?�\)���R�w�CxR                                    Bx��  �          @��Ϳ��H@J=q��33�I�B�R���H?����G��|�RC�                                    Bx�-z  �          @�������@R�\���C�B�\����@�
�����x�\B���                                    Bx�<   �          @��׿��@J�H��(��R�B��쿋�?�33���8RB�=                                    Bx�J�  �          @�Q쿆ff@U���R�H�
B��H��ff@
=��{  B��                                    Bx�Yl  �          @���\@C�
���H�D��B�{��\?�\)��Q��v(�Ch�                                    Bx�h  �          @�����\)@X����  �G�HB�aH��\)@	����  B�B���                                    Bx�v�  �          @��H��
=@R�\�����H  B��ῷ
=@33��  �}�RB��H                                    Bx��^  �          @�=q���\@+������e��B��Ϳ��\?�\)��{ffC                                      Bx��  �          @�\)�^�R@<(������_z�B���^�R?�33��z��B��H                                    Bx���  �          @�
=���@6ff��z��h�RB�33���?��
��
==qBѣ�                                    Bx��P  �          @����=q@&ff��p��r�
B�𤾊=q?����{� B��)                                    Bx���  �          @���<�@  ����B��<�?k���
= �B�                                    Bx�Μ  �          @��>�=q@!G������u�\B���>�=q?�(������qB��                                    Bx��B  �          @��R�\)@Q���z��{\)B�녾\)?�=q��33��B��                                    Bx���  �          @�
=��\)@����33�y��B����\)?�\)���B��                                    Bx���  �          @�33>8Q�@(�����z�
B���>8Q�?�\)��
=k�B��f                                    Bx�	4  �          @��H<#�
@,�����H�n  B���<#�
?���(��B�W
                                    Bx��  �          @�ff=#�
@)�������t
=B�.=#�
?������{B�W
                                    Bx�&�  �          @��H=�Q�@1G����
�q�B�Q�=�Q�?�z���p�B���                                    Bx�5&  �          @�\)<#�
@*�H��G��s\)B���<#�
?����=q��B�Q�                                    Bx�C�  �          @��>.{@'
=����w�B�W
>.{?�G������B�(�                                    Bx�Rr  �          @��H>\)@!G���\)�|�\B��R>\)?�33��
=\B�L�                                    Bx�a  �          @�Q��@#�
���
�y(�B�ff��?�(����
W
B�Q�                                    Bx�o�  �          @��>W
=@-p����\�r�B��>W
=?�\)�����B��\                                    Bx�~d  �          @�녿(�@Q�����33B���(�?Tz���p�Q�B���                                    Bx��
  �          @�33�=p�@ ����(��B�aH�=p�?0�����#�C�                                    Bx���  �          @��׾�G�?���ffz�B�8R��G�>�33���R§�RC�                                    Bx��V  �          @��;�p�?�p���Q�33B��þ�p�?#�
��33¤aHB��                                    Bx���  �          @��ÿ!G�@Q�����\B��H�!G�?E���{=qC.                                    Bx�Ǣ  �          @�ff���
@QG����
�I
=B��ῃ�
@
=��=q  B���                                    Bx��H  �          @��ÿn{@n�R�~{�7ffB�  �n{@&ff�����o�
B�W
                                    Bx���  
�          @��\�G�@n�R�����;
=Bˏ\�G�@%����
�t{BՏ\                                    Bx��  �          @�33�\@s33���\�;z�B�� �\@(������u�RB�z�                                    Bx�	:  �          @�����@w
=�����7��B�Ǯ���@.{���
�q�B�z�                                    Bx�	�  �          @��\��\@�  �vff�.�B ��\@9������h�B��                                    Bx�	�  �          @�=q�(�@�z��j�H�%
=B���(�@E����H�^�\B�u�                                    Bx�	.,  �          @�33�J=q@��S�
�ffB�#׿J=q@]p���=q�Jz�Bͨ�                                    Bx�	<�  �          @��
�Q�@�
=�R�\�Bȳ3�Q�@`  ����H�B�B�                                    Bx�	Kx  �          @�녿n{@����Z�H���B̔{�n{@R�\��(��PG�BӮ                                    Bx�	Z  �          @�=q��(�@��^{�Q�B�uÿ�(�@K�����QG�B��H                                    Bx�	h�  �          @�=q�z�H@��
�g
=�!�BΨ��z�H@E�����Y�B��                                    Bx�	wj  �          @�녿��@�p��Fff���Bԅ���@`  ��33�>��B�\)                                    Bx�	�  �          @��ÿ���@���%���33Bօ����@w��j=q�&�B�                                    Bx�	��  T          @�G���33@�33�6ff���B�\��33@`���u�/�B�q                                    Bx�	�\  �          @��
��(�@���dz����BԨ���(�@I����\)�T33B�\)                                    Bx�	�  �          @�����H@�  �,����z�B����H@l(��n�R�&�B��)                                    Bx�	��  �          @�����@��R�&ff��  B�\)���@k��g��!�\B�z�                                    Bx�	�N  �          @�����=q@����Tz��ffB����=q@E���R�F
=B�u�                                    Bx�	��  �          @�\)��z�@����U��p�B���z�@E���R�I  B�                                     Bx�	�  �          @�\)��z�@e�r�\�.�B�\��z�@#�
�����`��B���                                    Bx�	�@  �          @��;�ff@��R�Dz��  B��;�ff@tz���(��<\)B�Q�                                    Bx�
	�  �          @��
>#�
@�\)�)����ffB���>#�
@�p��z=q��RB�{                                    Bx�
�  �          @��H���
@���333����B�𤿃�
@�
=��Q��$��B��                                    Bx�
'2  �          @��H�G�@�ff�%��B�33�G�@����u��B��                                    Bx�
5�  T          @�33��\)@���)����Q�B�� ��\)@��z=q�Q�B�                                    Bx�
D~  
�          @��
<�@�=q�"�\��z�B�p�<�@����tz��Q�B�\)                                    Bx�
S$  
�          @��H��  @���{���HB�
=��  @��R�aG��\)B��q                                    Bx�
a�  �          @�Q�?�{@��\�˅�n�\B�?�{@����<����B��                                    Bx�
pp  �          @�33?:�H@��׿������B���?:�H@�p��HQ���33B�                                    Bx�
  �          @�p�<��
@��˅�p(�B���<��
@�(��>{��G�B��{                                    Bx�
��  �          @�p��
=q@�p�����vffB�aH�
=q@�33�@���陚B�z�                                    Bx�
�b  �          @Å=��
@���  ���\B�B�=��
@���aG����B�                                    Bx�
�  �          @\?
=q@����	�����B�\?
=q@�\)�[��	B���                                    Bx�
��  �          @Å?:�H@��
����33B��?:�H@���c33�(�B�                                      Bx�
�T  �          @�\)?G�@�����p���p�B���?G�@���S33�(�B��q                                    Bx�
��  �          @�\)>��@���A����B���>��@�G���
=�+B�8R                                    Bx�
�  �          @�
=?�@�p��XQ���B���?�@�\)��Q��:z�B�{                                    Bx�
�F  T          @�p�?��@�ff�.�R��p�B�k�?��@�����  �Q�B�#�                                    Bx��  
�          @�{?���@�\)�(Q����B��3?���@�ff�y���
=B�L�                                    Bx��  �          @�=q>���@��
�����{B��>���@�{�aG��G�B���                                    Bx� 8  �          @��
>���@������H��33B�aH>���@�p��Tz���p�B��R                                    Bx�.�  �          @��
�#�
@��
��R��G�B�G��#�
@�(��q���B�W
                                    Bx�=�  �          @�33=#�
@��\�!���G�B�.=#�
@��\�tz����B�\                                    Bx�L*  �          @ƸR����@�=q�E���(�B�#׾���@��R����-��B��                                    Bx�Z�  �          @��?���@�=q�
=���B�.?���@�p��Z=q�G�B���                                    Bx�iv  �          @�33?�
=@��!���(�B�p�?�
=@�{�qG��\)B�
=                                    Bx�x  �          @ʏ\>���@�\)�(Q�����B�� >���@�
=�xQ���RB��                                    Bx���  �          @ə�?\(�@�(��'
=��33B��{?\(�@�(��u�(�B��                                    Bx��h  �          @�G�?8Q�@����4z��ծB�.?8Q�@�  �����p�B��q                                    Bx��  �          @��
�k�@�{�x����\B��3�k�@j�H����Sz�B�B�                                    Bx���  �          @�33��z�@��\�mp���B��)��z�@vff���R�J�B��{                                    Bx��Z  �          @��;�
=@�Q��`  �ffB����
=@��\�����?�B��)                                    Bx��   �          @�33�n{@�  ��
=�0�\B��n{@J�H��=q�c�\BԳ3                                    Bx�ަ  �          @���aG�@��������8G�B̨��aG�@>{����k�B�8R                                    Bx��L  �          @�
=��G�@�ff��G��*��B�aH��G�@J�H��z��\G�B�B�                                    Bx���  �          @�����(�@I���B�\���C����(�@���j�H�ffC�R                                    Bx�
�  �          @�=q�l(�@|(��1���  C)�l(�@Mp��fff�G�C
�                                    Bx�>  �          @���
=q@�=q�\���	33B�\�
=q@k���(��7(�B�Ǯ                                    Bx�'�  �          @�����
@�(��\���	{B�  ��
@o\)�����7�B��q                                    Bx�6�  �          @��Ϳ�p�@�{�c�
��RB܅��p�@q������=z�B�B�                                    Bx�E0  �          @�z��G�@�{�j=q�ffB�Ǯ��G�@p������B�HB��f                                    Bx�S�  �          @�{���
@�(��_\)�	Q�B�녿��
@~�R����9�
B�(�                                    Bx�b|  �          @ƸR��\)@��
�e��B�aH��\)@|�����\�>ffB�.                                    Bx�q"  T          @�\)���\@��H�l���  B�LͿ��\@y�����C
=B��                                    Bx��  
�          @�p����@��qG����B�(����@o\)���R�F�B��f                                    Bx��n  �          @�����@����u��B�p�����@l(������I��B�B�                                    Bx��  �          @��Ϳ��@�Q��|��� \)B�k����@a���33�Pz�Bܳ3                                    Bx���  �          @�zῡG�@�33�w
=�(�BҞ���G�@h�������Lp�B�#�                                    Bx��`  �          @�����
=@���mp���Bֽq��
=@i������F��B�                                    Bx��  �          @�33��@�33�i����HB���@l(�����AffB�q                                    Bx�׬  �          @��
��\)@�\)�X����HB�(���\)@xQ����H�4B�u�                                    Bx��R  �          @Å��\@�G��`�����B�aH��\@k�����9G�B�
=                                    Bx���  �          @Å��@��R�{�� {B�=��@QG�����KB�Ǯ                                    Bx��  �          @�G���
@����n�R���B�\)��
@XQ�����D=qB�                                    Bx�D  �          @�  �\)@��
�k���B�(��\)@O\)��\)�B�B�z�                                    Bx� �  T          @\��@n{��{�.�B�(���@/\)��z��V��Cff                                    Bx�/�  �          @�����@hQ����R�0=qB�����@)����z��W33C&f                                    Bx�>6  �          @����R@g����R�2�RB�B���R@)����(��Z��C�                                    Bx�L�  �          @�
=��@��\�q���B�z��@j�H����N33B�G�                                    Bx�[�  �          @����E�@�33�w
=�ffB�
=�E�@k�����NG�Bˮ                                    Bx�j(  �          @�
=���
@�  �y���#33B۸R���
@U���ff�P(�B�k�                                    Bx�x�  �          @����@�
=�s�
�
=B�W
��@dz�����Lz�B�B�                                    Bx��t  �          @�Q쿓33@��
�|(��$p�Bр ��33@\�������R�RB��H                                    Bx��  �          @�ff���@�\)�|(��&Q�B�����@Tz���\)�S��B��                                    Bx���  �          @��׿���@�{�K�� �RB̀ ����@�p���z��/\)B��
                                    Bx��f  �          @�����@�����33�-33B֊=���@N{���
�Z=qB�                                    Bx��  �          @��H��(�@�z��@����33BθR��(�@����  �$�RB��                                    Bx�в  T          @Å���
@�{�[���
B�uÿ��
@�(����
�7�B��H                                    Bx��X  T          @�=q�fff@�ff�k��=qBɽq�fff@vff��=q�Dz�B�k�                                    Bx���  �          @�=q���@�  �G
=���RB�k����@�Q���=q�+\)B�W
                                    Bx���  �          @��
����@�=q�H������B�aH����@��\��33�(ffB�L�                                    Bx�J  �          @��
�n{@��������p�B�8R�n{@�p��^{�
��BɅ                                    Bx��  �          @����ff@�=q�;�����B�G���ff@���|(���B͔{                                    Bx�(�  �          @��H��
=@�p��@  ��B���
=@��R�~�R�#
=BѮ                                    Bx�7<  �          @Å��@�(��Dz���B�Ǯ��@����G��%�HB���                                    Bx�E�  �          @�p���{@���<(���B˔{��{@��
�|�����B�                                    Bx�T�  �          @���}p�@�  �&ff�ǅB�ff�}p�@��
�h�����B���                                    Bx�c.  �          @�\)��33@����%���z�B̊=��33@����e���BϮ                                    Bx�q�  �          @�����@�(��*�H��  B�Ǯ����@�  �hQ��{BԞ�                                    Bx��z  �          @�  ���
@�\)�%��B��)���
@��
�aG����BԮ                                    Bx��   T          @�33��ff@�{�.�R���B�  ��ff@���fff� ��Bׅ                                    Bx���  �          @��R��ff@�\)�E��	33B�.��ff@c33�vff�333B�#�                                    Bx��l  �          @�\)���R@�
=�H����B��)���R@a��y���6(�B�                                    Bx��  �          @��\��33@��\�?\)���B�#׿�33@Z�H�n�R�2  B��                                    Bx�ɸ  �          @�=q���
@���>{�G�B����
@]p��mp��1�B��)                                    Bx��^  �          @����33@w��S33�Q�B�#׿�33@J=q�\)�@��B�{                                    Bx��  T          @��\���R@]p��a��%�B����R@.{��(��K33B�aH                                    Bx���  �          @��H��\@Y���dz��(=qB��
��\@)������L�B�
=                                    Bx�P  �          @���;�@���p���8  C�=�;�?����33�N�C��                                    Bx��  �          @����7�@!��i���.ffC
���7�?������H�RC\                                    Bx�!�  �          @�z��)��@J=q�_\)�!
=C���)��@(������@C	Y�                                    Bx�0B  �          @���R@[��c33�#��B�(���R@,����(��F��C��                                    Bx�>�  �          @����\)@aG��g
=�#\)B����\)@1����R�F�RC �=                                    Bx�M�  T          @��\���@~�R�Q��
=B�{���@R�\�~{�5G�B�\                                    Bx�\4  �          @��H��\)@}p��\�����B�k���\)@P  ��(��?(�B�q                                    Bx�j�  �          @�z��   @w
=�e��
=B��f�   @G�����C33B�\)                                    Bx�y�  �          @�z��\)@�  �`  ���B����\)@Q����?�\B�W
                                    Bx��&  �          @����
=@����b�\�  B�녿�
=@U���\)�E��B�\                                    Bx���  �          @�Q�Q�@��\�Y�����B�=q�Q�@w����:=qB���                                    Bx��r  �          @�G��B�\@��
�Z=q�33Bƙ��B�\@z�H��ff�9�RB��)                                    Bx��  �          @��׿h��@����R�\�
��B�\�h��@~�R���\�4{Bͳ3                                    Bx�¾  �          @�Q�s33@�\)�Fff���B�Ǯ�s33@��H�z=q�+�HB�=q                                    Bx��d  �          @�\)��=q@���0����{B��)��=q@��\�fff���B�\                                    Bx��
  �          @��\���@����-p���B��쿇�@�\)�e��\)B�                                    Bx��  �          @��׿J=q@�
=�3�
��p�B�#׿J=q@�(��i���z�BȀ                                     Bx��V  
�          @�Q�W
=@�ff�4z���RB�G��W
=@���j=q��
B���                                    Bx��  T          @����Q�@�  �4z���RBƙ��Q�@�p��j�H��RB�                                      Bx��  �          @���p��@���4z���B�Ǯ�p��@��\�h���B̙�                                    Bx�)H  �          @�G��s33@�z��;���B�{�s33@����p  �!��B�
=                                    Bx�7�  �          @��\��Q�@����p���(�B��f��Q�@�  �Vff�p�B��q                                    Bx�F�  �          @�zᾸQ�@��\�{��{B�����Q�@�=q�W
=�=qB�u�                                    Bx�U:  �          @�(���G�@�z��z����B�=q��G�@���N{�{B�z�                                    Bx�c�  �          @����ff@���33��33B�z��ff@�(��L���ffB�k�                                    Bx�r�  �          @�33�E�@�G��
=��Q�Bę��E�@���O\)�p�B�=q                                    Bx��,  �          @�z�=p�@�������B�Ǯ�=p�@�(��Mp��(�B�L�                                    Bx���  �          @��5@���$z����HB�=q�5@�G��\(��p�B��)                                    Bx��x  �          @�=q�\(�@�(��8����B��H�\(�@�=q�k��\)B�ff                                    Bx��  T          @����@��H�l(��$z�Bѽq���@Y�����\�J  B�ff                                    Bx���  T          @�녿0��@����%���
B�W
�0��@�z��Z�H�p�B�                                      Bx��j  �          @��\�5@��
�(Q����Bó3�5@�33�\���G�B�k�                                    Bx��  �          @��H�J=q@�z��Fff� Q�B�z�J=q@����xQ��&G�B���                                    Bx��  �          @��ÿ8Q�@�Q��1���{B�k��8Q�@�
=�dz����B�W
                                    Bx��\  �          @����\@����p���G�B����\@��R�E���G�B�Ǯ                                    Bx�  �          @�G��\(�@�33�%��z�B�=q�\(�@�33�X����B�G�                                    Bx��  �          @��ÿ�  @�{�3�
���B�𤿀  @���e���B͙�                                    Bx�"N  �          @�  ��=q@�{�.�R����B̽q��=q@��`  �(�Bυ                                    Bx�0�  �          @�  �s33@��H�����Q�B�(��s33@�z��L(���B�33                                    Bx�?�  �          @�(�����@����H�pQ�B�L;���@��
�z��\B��                                    Bx�N@  �          @�Q��
�H@)���}p��D�\CW
�
�H?�(����\�_�\C	��                                    Bx�\�  T          @���Q�?�����  �e  C	���Q�?��H��  �z�
C��                                    Bx�k�  
�          @��ͿQ�@|(��U��B˙��Q�@W
=�{��B�BϏ\                                    Bx�z2  �          @��׾�p�@����*�H��33B�8R��p�@����Vff�  B�Q�                                    Bx���  �          @�  ��p�@Tz��_\)�2ffBܔ{��p�@.�R�~{�T{B�z�                                    Bx��~  �          @��\��G�@#�
�s33�P
=B����G�?�
=����n  C \                                    Bx��$  �          @��\����@��x���ZffB�W
����?����ff�vffCٚ                                    Bx���  �          @��Ϳ�@	���~{�Z=qCs3��?�G���Q��sz�C�
                                    Bx��p  �          @��Ǯ@������c{B�Ǯ�Ǯ?��
�����~Q�C�                                     Bx��  �          @����\)@Q��}p��Yp�CJ=��\)?�  ����r
=C\)                                    Bx��  �          @��R��\)?�ff��\)�iC  ��\)?�����ff�z�C��                                    Bx��b  �          @����?�����R�l=qCxR��>�(����\�wG�C)=q                                    Bx��  �          @��R����?�=q��ff�iC� ����?�����C!H                                    Bx��  �          @�ff��?������g��C���?�p���z��~(�C��                                    Bx�T  T          @��Ϳ��
@p���33�b�
B�p����
?�����(��}��CG�                                    Bx�)�  
�          @�  ����@<���@  �'��B�B�����@p��Z=q�Ez�B�(�                                    Bx�8�  �          @�ff��ff@���O\)�<�B����ff?�33�c�
�V�C�                                    Bx�GF  �          @����@(Q��N�R�=�B�z῱�@Q��e�Z�HB�(�                                    Bx�U�  �          @�\)��{?�33�c33�T��Cc׿�{?�{�r�\�k
=C��                                    Bx�d�  �          @��ÿ��@p��aG��U�B�=q���?��s33�q  C�                                    Bx�s8  �          @��\�c33?�(��#�
��HC!
=�c33?W
=�-p��(�C&�                                     Bx���  �          @����@��?�ff�A��'��C�\�@��?���N�R�4��C {                                    Bx���  �          @����N�R?xQ��>{�%��C#Q��N�R?��E��-{C*�                                    Bx��*  �          @����@��?��R�I���0G�C���@��?G��S33�:�RC%}q                                    Bx���  �          @�  �8��?�G��L���5��Cz��8��?J=q�U�@�\C$��                                    Bx��v  �          @����3�
?����Q��9�RC\)�3�
?^�R�[��E��C"                                    Bx��  �          @��,��?��L���9\)C5��,��?u�W��FffC p�                                    Bx���  
�          @����(��?�ff�K��@C@ �(��?���S33�JQ�C'8R                                    Bx��h  T          @�Q��*�H?�ff�XQ��B�C)�*�H?Q��a��NffC#
=                                    Bx��  �          @�\)�=q?��H�\���KQ�C���=q?xQ��g��Y�\C&f                                    Bx��  �          @�  �"�\?�\)�Z�H�Gz�C���"�\?c�
�dz��TG�C ��                                    Bx�Z  �          @�ff��R?����y���d
=C����R?(�������p{C#�\                                    Bx�#   �          @����
=?�G���\)�y�
CxR�
=>Ǯ��=qǮC)�=                                    Bx�1�  �          @�\)��?xQ���z��{G�C��>�����\)\C+J=                                    Bx�@L  �          @���:�H?�(���z��N�RCJ=�:�H?fff�����ZG�C"ٚ                                    Bx�N�  T          @���W�?�z��a��)\)C��W�?��p  �6�C0�                                    Bx�]�  �          @�Q��\(�?��a��(�
CL��\(�?����n�R�5p�C��                                    Bx�l>  �          @����b�\@ff�R�\��RC\)�b�\?���b�\�(�HC{                                    Bx�z�  �          @���Z�H@�\�`���%�\C+��Z�H?Ǯ�p  �3��Cn                                    Bx���  �          @��H�I��?�
=�u�8Q�C�\�I��?�z�����FG�C޸                                    Bx��0  
�          @�33�>�R?������EC�R�>�R?�  ��Q��S=qC5�                                    Bx���  T          @��
�:�H?�\��z��I�RC�=�:�H?�p����\�W(�CB�                                    Bx��|  �          @�33�5?�=q�����G��C&f�5?�ff��
=�U��CQ�                                    Bx��"  �          @�z��,��?��\���`=qC�R�,��?333����j33C%k�                                    Bx���  �          @�{�p�?����33�}(�C�f�p�>�
=��{{C)W
                                    Bx��n  �          @�{�
=?��R��33�}  C�{�
=?�R���RB�C#��                                    Bx��  �          @����/\)?O\)�����gp�C#��/\)>u���
�l��C/                                    Bx���  �          @��������\C7�f���@  ����ǮCJ5�                                    Bx�`  �          @�{��녾B�\��p��=C:���녿Tz����k�CN��                                    Bx�  �          @����׽�Q����.C6����׿5��Q�Q�CH��                                    Bx�*�  T          @�33�  =���=q�
C0���  �������ǮC?��                                    Bx�9R  �          @�녿�\)�#�
��{k�C533��\)�!G�����CF��                                    Bx�G�  T          @���� �׾B�\����C9c�� �׿E�����ǮCH��                                    Bx�V�  �          @���������
=�|�C6������.{��p��w��CCٚ                                    Bx�eD  �          @�=q��\)��G���ff��CAW
��\)���\����3CP�R                                    Bx�s�  �          @���1G���  ��{�g�RC9\�1G��B�\��(��b�CCff                                    Bx���  �          @�\)���\���
�w�
C<����k���G��p��CH��                                    Bx��6  �          @��R��;���=q�u��C>�{��Ϳz�H����n�CI�
                                    Bx���  �          @��C�
��33����V{C:z��C�
�Q����H�Q(�CC
=                                    Bx���  �          @���Dz�����33�S
=C>
�Dz῁G���Q��L��CF33                                    Bx��(  �          @�(��I�������Q��N(�C<���I���k��{��H�\CDQ�                                    Bx���  �          @�=q�b�\�8Q��g
=�5�
C6��b�\�z��c�
�3{C=\)                                    Bx��t  �          @�(��)����=q��(��jC9�)�)���B�\��=q�e�CC��                                    Bx��  �          @�33�:=q=#�
��p��\�C3&f�:=q��
=�����Zz�C<(�                                    Bx���  
�          @���.�R=�G�����f�C1���.�R��Q������effC;�                                     Bx�f  �          @��\�=q��p���ff�t��C<���=q�\(���(��n��CG�=                                    Bx�  �          @�Q������������j�
CLaH�����
��(��^�CT�)                                    Bx�#�  �          @�G���G�?��H���ffB͏\��G�?�  ���HQ�B���                                    Bx�2X  �          @���=�G�@
=q���H�x��B�\)=�G�?��H����� B��H                                    Bx�@�  �          @�p�>.{�}p����8RC���>.{���H��33C�`                                     Bx�O�  �          @��\>L��?��
��{.B�#�>L��?�ff��33�B��                                     Bx�^J  �          @�zἣ�
?���z���B�.���
?.{���£�qB��q                                    Bx�l�  �          @��ͽ�G�?У�����B��׽�G�?�33��Q��qB��
                                    Bx�{�  
�          @��ͽL��?�G������B�k��L��?��
��G��RB�                                    Bx��<  �          @��
��?��H��ff\B�\)��?8Q�����¢��B��)                                    Bx���  �          @��\���
?&ff����¤\)B��켣�
>.{���¯�fB�                                    Bx���  
(          @�z὏\)?Y������fB��὏\)>�Q����
«L�BȽq                                    Bx��.  T          @��ͼ��
?W
=��=q L�B��q���
>�33��(�«�B�u�                                    Bx���  �          @�(���\)?�z���\)�RB�Q쾏\)?0����=q¢� B��\                                    Bx��z  �          @���&ff?�=q���H=qB����&ff?�\)��\)Q�B��                                    Bx��   �          @��׿^�R=L������C0���^�R������z���CL                                    Bx���  �          @���z�H�(����(�CVB��z�H�����33�)Cd�                                    Bx��l  �          @�zῌ�;�����
=��CH����ͿQ����\)CX��                                    Bx�  �          @�zῘQ콸Q���\){C8xR��Q��\��ffp�CKW
                                    Bx��  �          @�������
����fC7�ÿ���   ���R�=CIO\                                    Bx�+^  �          @����p�>B�\��=q�C+(���p��k���=q\)C>�R                                    Bx�:  �          @�녿��\?E����\C�ÿ��\>�{��(�.C$�                                    Bx�H�  �          @�  ��p�?��R�����3C����p�?W
=���k�C��                                    Bx�WP  �          @�Q쿵?E���ff�Cff��>�p���Q���C%n                                    Bx�e�  
�          @�Q쿴z�?�ff���k�CY���z�?&ff��\)�3C.                                    Bx�t�  �          @����\)?�G���Q�ǮB�B���\)?�����(�u�Cz�                                    Bx��B  �          @�\)���R@�
��33�t�B�����R?���Q�p�B�.                                    Bx���  �          @�  ��
=?�{��z��w��B��
��
=?�(���G�ffC
=                                    Bx���  �          @��Ϳ�z�>�
=��ff��C%�)��z�<���
=L�C2޸                                    Bx��4  �          @��H�
=���
��ff�rz�C6�
=��G���p��pQ�C>�{                                    Bx���  �          @��H���p����\)�u  CK�{�������z��l  CSQ�                                    Bx�̀  �          @��׿��R�E���(��COxR���R��������CYff                                    Bx��&  �          @��8Q쾮{���\ Q�CMO\�8Q�8Q�����Ca�                                    Bx���  �          @�p�����k������{C@����������\)�CR
                                    Bx��r  �          @�(��fff>8Q���Q���C(�3�fff�B�\��Q��C?�q                                    Bx�  �          @�z��\>.{���H¦��C!c׿�\�L�����H¦ffCI:�                                    Bx��  �          @�p����
��  ��(�©�CZ�)���
��R���H¢�3Cr��                                    Bx�$d  
�          @�z�\������H¤��Cf���\�Tz���G�=qCu.                                    Bx�3
  
�          @��\�333��ff�����CT�3�333�L����ff
=Cd�                                    Bx�A�  �          @��H��  ��  ����«\Ca�þ�  �����  £� Cw�
                                    Bx�PV  �          @�{�333��G���33��CTh��333�J=q�����fCd�3                                    Bx�^�  �          @����33�aG���\)#�CY+���33���H����W
Cbff                                    Bx�m�  �          @���=L�;�
=���©ǮC�K�=L�ͿE���{¡8RC�˅                                    Bx�|H  �          @�G���=q?���\)¤�qB����=q>aG���Q�«��Cs3                                    Bx���  �          @�G��W
=>�\)����«�B��H�W
=���
����®��CG��                                    Bx���  �          @����#�
��33����«��C���#�
�0����\)£Q�C�u�                                    Bx��:  �          @��>L��=�����Q�®��A�z�>L�;u��Q�¬�\C��{                                    Bx���  �          @��R>��?�\����¦�\Bz�>��>#�
��p�¬A��R                                    Bx�ņ  �          @��\��Q�>\)��=q¯��B���Q�8Q���=q®�HCr�
                                    Bx��,  
�          @����G��333��G�¡=qC�o\��G���  ��\)\)C�Ǯ                                    Bx���  T          @��H�녿�ff�w���Cqz�녿���r�\�RCv��                                    Bx��x  �          @�ff�����.{�}p���CQ�쿙���s33�y��L�CZQ�                                    Bx�   �          @�{�E��+�����ǮC\�R�E��p����  =qCf�f                                    Bx��  �          @�z���Ϳ333��G�k�Co�
���Ϳu�~�R��Cw0�                                    Bx�j  �          @�33=�\)��\)�|(�Q�C���=�\)��\)�vff��C�\)                                    Bx�,  �          @��=��
�aG����
�RC�y�=��
��33����Q�C��                                    Bx�:�  �          @���<��Y����{C��<���{����qC���                                    Bx�I\  �          @��R>����L�����
u�C�L�>����������{C��\                                    Bx�X  �          @���>�ff�   ��
=¢��C��\>�ff�G���  C��                                    Bx�f�  �          @��H>��R��R����¢u�C�g�>��R�fff��\)��C���                                    Bx�uN  �          @���>��+���\)33C�>��p����=qC�w
                                    Bx���  �          @�=q>�
=�J=q��\)k�C�33>�
=��ff��p�.C��q                                    Bx���  �          @���>���\(���{.C�U�>����\)��(���C�w
                                    Bx��@  �          @�G�=�G��k���{z�C��H=�G���
=��(���C��q                                    Bx���  �          @�>aG���G����\)C�q>aG����
����)C���                                    Bx���  �          @�Q�?��R������\(�C��\?��R���
��  �}\)C�{                                    Bx��2  �          @�Q�?z῜(������fC���?zῼ(���
=��C���                                    Bx���  �          @�\)?h�ÿ�{����C���?h�ÿ������{C��                                    Bx��~  �          @��H?.{��{���H\)C�XR?.{��{��Q��C���                                    Bx��$  �          @�  ?   ���������HC�  ?   �˅��ffC��\                                    Bx��  �          @�{>��
��=q���(�C�>��
�Ǯ���8RC���                                    Bx�p  �          @�  ?+�������H��C�w
?+����
������C���                                    Bx�%  �          @��
?J=q��ff��z�C��?J=q���\���\�HC��3                                    Bx�3�  T          @���?��E��{�W
C��\?��xQ��x��L�C��f                                    Bx�Bb  �          @��\?�(��.{�z=qffC�C�?�(��^�R�w��~{C���                                    Bx�Q  �          @�\)@�
���
�mp��gp�C�f@�
��\�l(��ez�C��{                                    Bx�_�  �          @��@���?�
=���
���RAw33@���?��Ϳ���ffAe��                                    Bx�nT  �          @�{@��
?����33���AK33@��
?xQ쿹����(�A>=q                                    Bx�|�  �          @�{@�(�?^�R���R��A!G�@�(�?L�Ϳ��
��\)AQ�                                    Bx���  �          @���@���?�=q�k��$z�A���@���?��
�}p��1��A�                                    Bx��F  �          @�@��
?�녿���g�ARff@��
?�=q����p��AG�                                    Bx���  �          @���@�  ?^�R������{A'
=@�  ?O\)���R����A�\                                    Bx���  �          @��@�?z�H��
=���A<��@�?h�ÿ�(����A0��                                    Bx��8  �          @���@��\?�R�����r�R@�@��\?녿����w�@�                                    Bx���  �          @�{@�G�?�G�?O\)A�A��H@�G�?��?=p�A�A��                                    Bx��  �          @��@�ff?��R?\(�A\)A��@�ff?\?L��A(�A���                                    Bx��*  �          @�G�@�  ?���?��
A3
=Avff@�  ?�?xQ�A(��A}G�                                    Bx� �  �          @�@�\)?�p�?@  Ap�A^{@�\)?�G�?333@���Ac33                                    Bx�v  �          @���@��?��R>�ff@��HAd��@��?�  >���@���Ag�                                    Bx�  �          @�@���?��\>�
=@�{A@��@���?��>\@�
=AC�                                    Bx�,�  �          @�p�@��\?Q�>�=q@J=qAG�@��\?Tz�>u@333A
=                                    Bx�;h  �          @�{@�=q<��
����F�R>aG�@�=q���
����F�RC��                                     Bx�J  �          @�ff@��׿�z���\��C��)@��׿��R�  ��RC�q�                                    Bx�X�  �          @��@���  �����
C��q@�����У���Q�C��                                    Bx�gZ  �          @�p�@�Q쿏\)�
=���HC��q@�Q쿙���z��ȣ�C�'�                                    Bx�v   �          @�p�@�ff?E�?���AJ�RA  @�ff?L��?��AEp�A�H                                    Bx���  �          @�G�@�=q?�Q�?��RA�p�A���@�=q?�G�?�
=A�  A�ff                                    Bx��L  �          @��@���?�?��A�ffA��@���?�z�?�A��RA�=q                                    Bx���  �          @��@�=q?��H?�G�A��
A��@�=q?\?��HA�p�A�                                      Bx���  �          @�G�@�ff?��?�(�A�33AA�@�ff?���?�
=A|Q�AI�                                    Bx��>  �          @�ff@�33?&ff?�ffA��R@�33@�33?0��?��
A���Aff                                    Bx���  �          @���@��׿��R�\)��G�C�j=@��׿�  ��\��
=C�Q�                                    Bx�܊  �          @�z�@`���@  ������{C�@`���B�\��p���G�C���                                    Bx��0  �          @�@Dz��hQ쿢�\�nffC��@Dz��j�H��z��Z=qC��)                                    Bx���  �          @�ff@0����  �aG��#�C�H�@0�����׿B�\�ffC�4{                                    Bx�|  �          @�z�@���
=?\(�A(��C�aH@���?k�A4Q�C���                                    Bx�"  �          @��
@��\@ ��@�A�Q�A�  @��\@z�@�A��HA�33                                    Bx�%�  �          @�p�@n�R@"�\@�A�RB��@n�R@&ff@�A�  B�\                                    Bx�4n  �          @��@���@��@\)A�33A�G�@���@ ��@�A��B 33                                    Bx�C  �          @��@o\)@
=q@c�
B�A�Q�@o\)@  @`  B
=A�(�                                    Bx�Q�  �          @���@`  @33@r�\B,(�A�R@`  @��@o\)B)\)A�p�                                    Bx�``  �          @�33@[�?��R@|��B3�RA�R@[�@�@z=qB1  A��                                    Bx�o  
�          @��@Z�H?�@���B8�HA�  @Z�H?�
=@�Q�B6ffA�G�                                    Bx�}�  �          @�=q@J=q?��@���BH��A�p�@J=q?�p�@��BF��A�                                      Bx��R  �          @�G�@%�?�G�@��HB]�RA��
@%�?���@��B[33A�                                    Bx���  �          @��@ ��?���@��
Btp�B�@ ��?�z�@��HBq��B{                                    Bx���  �          @���@z�?�G�@���Bi=qA���@z�?���@�  Bg  A�p�                                    Bx��D  �          @�G�?\?���@���B���BQ�?\?�(�@�(�B�G�B�R                                    Bx���  �          @�ff@�?W
=@�{B~��A��@�?k�@�p�B}�A�
=                                    Bx�Ր  �          @�ff?���?G�@���B���A�=q?���?\(�@���B���A��                                    Bx��6  �          @�(�@*=q?��@�ffB[G�A�  @*=q?�{@�p�BY�\A��                                    Bx���  �          @��
@,(�?��@���BX�HA�\)@,(�?���@�(�BW33Aأ�                                    Bx��  �          @��\@)��?��
@�z�BZQ�A�p�@)��?��@��
BX�RA�z�                                    Bx�(  �          @�
=@#�
?�{@�G�BY�Aߙ�@#�
?�@���BW�HA�{                                    Bx��  �          @��@�?��H@u�BP�B�R@�?�\@s�
BO  B(�                                    Bx�-t  �          @�ff@.�R?�z�@q�BGffA�=q@.�R?��H@p��BEB G�                                    Bx�<  �          @�33@Fff?�  @s33B?Q�A�Q�@Fff?�ff@q�B>  A�=q                                    Bx�J�  �          @��@'
=@�@fffB<p�B
=@'
=@�@dz�B:�Bp�                                    Bx�Yf  �          @�z�@1G�@ ��@]p�B5�B��@1G�@33@[�B3�B                                    Bx�h  �          @�(�@6ff@@W
=B.\)B(�@6ff@Q�@UB,�
B{                                    Bx�v�  �          @�@:�H?��H@b�\B9\)A��@:�H?޸R@a�B8�A���                                    Bx��X  �          @�  @B�\?��
@i��B<=qA�@B�\?���@h��B;33A�{                                    Bx���  �          @�G�@Dz�@G�@Q�B"�
B��@Dz�@33@P��B!��BG�                                    Bx���  �          @�z�@Dz�@   @QG�BffB  @Dz�@!�@P  B�B=q                                    Bx��J  �          @��
@B�\@G�@[�B(ffB�@B�\@33@Z=qB'Q�B=q                                    Bx���  
�          @��H@C33@��@XQ�B&�\Bff@C33@�\@W
=B%�\B��                                    Bx�Ζ  �          @���@H��@ff@H��B�BG�@H��@�@HQ�BBG�                                    Bx��<  �          @�
=@L��@�H@>�RBp�Bff@L��@(�@=p�B��B=q                                    Bx���  �          @���@n�R@�R?�p�A��B�@n�R@\)?�(�A�B��                                    Bx���  T          @�(�@h��@1�?�A�Q�B�@h��@2�\?�z�A���B�
                                    Bx�	.  �          @��\@j�H@&ff?޸RA��Bff@j�H@'
=?�p�A�z�B�                                    Bx��  �          @�p�@p��@*�H?ٙ�A�=qB�R@p��@+�?�Q�A�G�B��                                    Bx�&z  �          @�  @mp�@.{?�
=A��B
=@mp�@.�R?�A��HBG�                                    Bx�5   �          @�=q@dz�@.�R?�
=A��
B�@dz�@/\)?�A�33B�
                                    Bx�C�  �          @��@hQ�@.�R?�ffA
=B@hQ�@.�R?��A~{B�H                                    Bx�Rl  �          @�p�@mp�@!G�?��A��RB��@mp�@!G�?��A�z�B�                                    Bx�a  �          @�Q�@s�
@"�\?�  AuG�B�@s�
@"�\?�  Au�B�\                                    Bx�o�  T          @�(�@�Q�@#�
?�=qAM�Bp�@�Q�@#�
?�=qAM�Bp�                                    Bx�~^  T          @�(�@�33@!�?8Q�A��A���@�33@!�?:�HA��A��                                    Bx��  �          @�(�@xQ�@(Q�?��\ArffB�@xQ�@(Q�?��\As
=Bz�                                    Bx���  �          @��\@~�R@"�\?�  A?�Bp�@~�R@"�\?�G�A@��B\)                                    Bx��P  �          @�  @�  @{?+�A ��A��@�  @{?+�A{A�\)                                    Bx���  �          @�Q�@��
@>�
=@���A�z�@��
@>�
=@��A�Q�                                    Bx�ǜ  �          @�ff@~{@���G���{A�p�@~{@���(����\A���                                    Bx��B  �          @��@xQ�@Q�=p��z�A�@xQ�@Q�:�H�ffA�                                      Bx���  �          @�{@��H@�;�(����A��@��H@p���
=���A���                                    Bx��  �          @��@�{?�G���{���
A���@�{?�G�������Q�A�33                                    Bx�4  �          @�33@�z�?�G���  �p��A��@�z�?�G����R�n�HA�=q                                    Bx��  �          @�33@h��?����&ff�(�A�G�@h��?�{�%�z�A���                                    Bx��  �          @�  @^{@33�:�H��A�Q�@^{@z��:=q�{A�ff                                    Bx�.&  �          @���@H��?�p��`���5(�A�  @H��?�G��`  �4ffA�33                                    Bx�<�  �          @��@E?�33�p  �BA�\)@E?�
=�o\)�B{A�\)                                    Bx�Kr  �          @�ff@B�\>\�xQ��N��@�@B�\>�
=�xQ��Nff@��                                    Bx�Z  �          @���@1�?����z=q�O\)A�@1�?�{�y���Np�AиR                                    Bx�h�  �          @���@��?\���R�d�B�R@��?Ǯ��{�c33B
��                                    Bx�wd  �          @��
?�z�?��\�����wz�B?�z�?��������v�B
p�                                    Bx��
  �          @�?��?�����
B�?��?�����33B!{                                    Bx���  �          @��?�\?aG����H�{A�  ?�\?p�����\��A߅                                    Bx��V  �          @�p�?�=q>\����G�A<��?�=q>�G�������AXz�                                    Bx���  �          @��?��
=���G��@mp�?��
>8Q���G�  @��                                    Bx���  �          @���?�?8Q����(�A�G�?�?G���33��A���                                    Bx��H  �          @��\?���>�����ff8RA-��?���>�p���ff�fAS�
                                    Bx���  �          @�
=@  �(���{�=C�h�@  �
=q��ff��C�B�                                    Bx��  �          @�ff?�ff��
=��33.C�~�?�ff��{����C��f                                    Bx��:  �          @�  ?�(�>�33�����AVff?�(�>�(����\)A��R                                    Bx�	�  T          @��H?W
=?����\)�3B33?W
=?0����
=�{B(�                                    Bx��  �          @�  ?E�>��H��p� ǮB=q?E�?z�����3B                                      Bx�',  �          @���?��H>�����(�Ar{?��H>�
=���
33A��\                                    Bx�5�  �          @�Q�@ ��>�(���3333AB�R@ ��?���33�qAiG�                                    Bx�Dx  �          @�  @�=�G������@.{@�>L�������@��                                    Bx�S  �          @�=q@G�=������\��@(��@G�>L�����\�@�                                      Bx�a�  �          @�Q�@Q�>�33������A=q@Q�>�ff��G�B�A>{                                    Bx�pj  �          @�
=?�\?.{���L�A��R?�\?G���33p�A��R                                    Bx�  �          @��?   >#�
���¨ǮA�
=?   >������§�fA�{                                    Bx���  �          @��H>�Q����G�¥� C�@ >�Q������§k�C�                                      Bx��\  �          @���?.{����
=¡��C�U�?.{��p���\)£�C��f                                    Bx��  
�          @�33?h�þ�
=��  Q�C��H?h�þ��R��Q�B�C���                                    Bx���  �          @��
?�G���
=��Q�u�C���?�G���������Q�C��H                                    Bx��N  �          @��?n{�\)��G���C�e?n{��G�����
=C�=q                                    Bx���  �          @�?�������(�¦  C�U�?���\)��z�§k�C��                                    Bx��  �          @�(�>��H��\���\¤�
C��q>��H�\���H¦��C��)                                    Bx��@  �          @�p�?J=q�����H L�C�aH?J=q��33��33¡�\C�H                                    Bx� �  �          @�ff?u�����H�C�0�?u�������H�{C���                                    Bx� �  �          @��?s33���R��z�C���?s33�8Q������qC���                                    Bx�  2  �          @��R?c�
��33���
�)C�T{?c�
�W
=���
 �qC�T{                                    Bx� .�  �          @�{?fff�+���=q.C��?fff�����\��C�ٚ                                    Bx� =~  �          @�\)?0�׿�  ���HffC�B�?0�׿\(�����
C�W
                                    Bx� L$  �          @�  ?B�\����G�u�C��?B�\���
�����C�K�                                    Bx� Z�  �          @��ÿJ=q�$z�����x�C|�3�J=q����p��~�HC|�                                    Bx� ip  �          @��H�!G��(���Q��
C�)�!G���\���\�fC~�q                                    Bx� x  �          @�{�\�(������
C�� �\��\���RC�W
                                    Bx� ��  �          @�z὏\)�����(��HC�(���\)�  ��ff(�C��                                    Bx� �b  �          @�����H�Q���#�C�� ���H��(����aHC�q                                    Bx� �  �          @�G����R�Q���  �C�AH���R�{��=qB�C���                                    Bx� ��  �          @��ÿ   �33������C��{�   ������\#�C�b�                                    Bx� �T  �          @�G��J=q�   �����{�C|s3�J=q�ff��
=G�C{Y�                                    Bx� ��  �          @�{>u�{��(��C���>u�33��ff
=C��)                                    Bx� ޠ  �          @���?z�H�	������ǮC�7
?z�H��(����HC�&f                                    Bx� �F  �          @�=q?�33�������{C���?�33��(����\C��\                                    Bx� ��  �          @��?\�z������C��?\��33�����qC�U�                                    Bx�!
�  �          @���?������Q���C�n?����33���\�RC���                                    Bx�!8  �          @��?�=q�������
�qC�޸?�=q����p��qC�T{                                    Bx�!'�  �          @��H?�����R����C�� ?����ff��
=
=C��{                                    Bx�!6�  �          @�=q?\(�����z�L�C�o\?\(������R�
C�|)                                    Bx�!E*  �          @���?��������C�K�?��   ���3C���                                    Bx�!S�  �          @Å?����p����HW
C�1�?����G�����
C�E                                    Bx�!bv  T          @�p�?��ÿ��R��p�C��R?��ÿ����  �RC�Ff                                    Bx�!q  
�          @�ff?��
���R��\){C�]q?��
�������p�C���                                    Bx�!�  
�          @Ǯ?�녿޸R��  �)C���?�녿��
���C�w
                                    Bx�!�h  �          @�\)?����H�����3C���?���G���=q(�C���                                    Bx�!�  T          @�\)?޸R���H��  �qC��f?޸R���R����{C���                                    Bx�!��  �          @�  ?�녿�=q��  W
C��?�녿�{����C�,�                                    Bx�!�Z  �          @�(�@ff������33�C��\@ff��p�����8RC��\                                    Bx�!�   T          @���@
�H��
=���H  C�Y�@
�H���H��z�8RC�z�                                    Bx�!צ  �          @�\)@(���z����\k�C��)@(���Q���(�u�C�f                                    Bx�!�L  �          @��@%���{��Q�C�W
@%��c�
������C�|)                                    Bx�!��  �          @�(�@6ff���
���
�v�C�3@6ff�O\)�����y=qC��                                    Bx�"�  �          @�(�@;����������r33C��3@;��aG����H�u{C��)                                    Bx�">  T          @�  @@�׿��H��z��pffC�\@@�׿z�H���s�C��                                    Bx�" �  �          @���@G�������H�kz�C��=@G������z��n�C��
                                    Bx�"/�  �          @���@J�H��ff�����i33C�ٚ@J�H��=q��33�lp�C��H                                    Bx�">0  �          @У�@W
=������p��`��C��@W
=������
=�d  C��3                                    Bx�"L�  T          @Ϯ@hQ�\����R��C��q@hQ쿧���
=�V{C��                                    Bx�"[|  �          @�G�@^{��{��\)�T{C��@^{��������XQ�C�^�                                    Bx�"j"  T          @��@L�Ϳ�������_(�C�%@L�Ϳ�������c�C���                                    Bx�"x�  �          @���@Y�����H���
�Wp�C�H@Y����p���ff�\
=C�|)                                    Bx�"�n  �          @���@U�����33�V�HC��@U��������[�
C�w
                                    Bx�"�  �          @�  @<(���������mQ�C�T{@<(������(��r�C�*=                                    Bx�"��  �          @أ�@;����H��33�o�\C��@;�������p��t33C���                                    Bx�"�`  �          @�G�@ff�Ǯ��p��
C�5�@ff�����\)k�C���                                    Bx�"�  �          @�33@����{�ƸR�C��@������ȣ��qC�c�                                    Bx�"Ь  �          @��
@!녿Ǯ��{C�&f@!녿����  =qC��H                                    Bx�"�R  �          @�=q@*�H��33�Å�~��C�#�@*�H��������{C��                                    Bx�"��  �          @��@:=q��{��  �w�C�|)@:=q�������{z�C��q                                    Bx�"��  �          @��H@=p���(���{�s�RC��@=p�������  �w��C���                                    Bx�#D  �          @�33@Tz��   ��z��^{C�xR@Tz�޸R��
=�c(�C�'�                                    Bx�#�  �          @љ�@33������r�
C�#�@33��Q������y��C��                                    Bx�#(�  �          @�33?����+���z��{=qC�*=?�����H��  p�C�=q                                    Bx�#76  �          @��>��H�6ff���R�|�\C���>��H�%����\z�C�S3                                    Bx�#E�  T          @�(�>B�\�HQ������r33C��3>B�\�7���{�|�
C���                                    Bx�#T�  �          @У׾.{�\�������g�C�� �.{�L(���p��rffC���                                    Bx�#c(  �          @�
=���^�R���d�C�;��N{���H�o33C��q                                    Bx�#q�  �          @�\)�����Tz������l{C�l;����C33��ff�v�HC�4{                                    Bx�#�t  �          @�(���
=�Y�������Y�Cw=q��
=�I������c�Cu�f                                    Bx�#�  �          @��ÿ�ff�Z�H����](�C|�׿�ff�J�H��=q�g��C{��                                    Bx�#��  �          @�{�!��aG������2G�Cj:��!��S�
���;�Ch��                                    Bx�#�f  �          @�(��S�
�j�H�fff�(�Cc��S�
�_\)�qG��Q�Cb��                                    Bx�#�  �          @��H�#�
�o\)��=q�'�
Ck���#�
�a���  �0��Cj�                                    Bx�#ɲ  �          @��H�Tz��h���c33�G�Cc�f�Tz��]p��n�R�z�Cb5�                                    Bx�#�X  �          @����
�g��G����C]G����
�]p��S33��C\�                                    Bx�#��  �          @�{��(��j�H�.�R����CX���(��a��:=q��ffCW�)                                    Bx�#��  �          @�=q��
=�p���1G����CY)��
=�g��=p��ͅCX\                                    Bx�$J  �          @�  ��{�hQ��3�
���CXJ=��{�^�R�?\)�ҏ\CW.                                    Bx�$�  �          @�ff����n�R�2�\���HCYǮ����e�>{����CX��                                    Bx�$!�  �          @���p��i���)����
=CX�)��p��`���5��ȣ�CW��                                    Bx�$0<  �          @����Q��w��%��33C[!H��Q��o\)�1��ŮCZ!H                                    Bx�$>�  �          @�(�����g��*=q��CX������^�R�6ff�˙�CW�
                                    Bx�$M�  �          @�(������g
=�%���CZk������^�R�1G���(�CYY�                                    Bx�$\.  �          @Ϯ������p���p���  C^�=������=q�(����
C]��                                    Bx�$j�  �          @�Q���=q��33�
�H���C]޸��=q�~�R����33C]�                                    Bx�$yz  �          @�����p����R�ff��
=CH)��p���{�������CF�                                    Bx�$�   T          @ƸR�����  ��
����C@��������������C?h�                                    Bx�$��  �          @���{�Q��(���G�CIaH��{�G������
CHff                                    Bx�$�l  �          @�
=����z�������
CK:�����{��(����CJQ�                                    Bx�$�  �          @��H��������
=�w�CE�f������  ���
��
=CD�{                                    Bx�$¸  �          @ə���(��Q�&ff��CG޸��(���B�\��ffCG��                                    Bx�$�^  �          @�(������A녿�\)�$Q�CQO\�����>{���
�<  CP�=                                    Bx�$�  �          @�ff��
=�J�H�����HCSJ=��
=�C�
�(���\)CRaH                                    Bx�$�  �          @��
��z��8Q��\��33CP&f��z��1녿�����COT{                                    Bx�$�P  �          @����=q�:�H�z���ffCP�q��=q�333�{��  COǮ                                    Bx�%�  �          @�G�����:=q�p����HCQ�����2�\������CP��                                    Bx�%�  �          @ȣ�����E�p���33CS����=p�����{CR�R                                    Bx�%)B  �          @��
����Fff�{���CS:�����>�R������
CR0�                                    Bx�%7�  �          @�������P  ����CT:�����H�������CS\)                                    Bx�%F�  �          @�{�����\(�� �����CVG������U��(����CUaH                                    Bx�%U4  �          @љ���ff�S33������HCTn��ff�K�����=qCSn                                    Bx�%c�  T          @�����]p��%����CX&f����S�
�0����Q�CV�R                                    Bx�%r�  T          @ȣ����R�`  �*�H���CZ
���R�Vff�7
=��G�CX�)                                    Bx�%�&  �          @�{�j=q�z�H�c�
�Q�Cb�R�j=q�mp��qG��z�Cas3                                    Bx�%��  �          @�G��`  �\)�s�
��\Cd�3�`  �qG������  Cc)                                    Bx�%�r  �          @�Q��S�
��33�s�
�\)Cg
=�S�
�xQ������33Ce�                                     Bx�%�  �          @�\)�i�������c33���Cc���i���tz��p�����CbT{                                    Bx�%��  T          @�
=�vff�~�R�U����Ca�f�vff�q��c�
��\C`xR                                    Bx�%�d  T          @�Q��k���Q��c�
���Cch��k��s33�qG��  Ca�f                                    Bx�%�
  �          @�
=�i���z=q�h����HCc  �i���l���vff�33Cak�                                    Bx�%�  �          @�\)�S�
��Q��w
=�\)CfxR�S�
�r�\���\�33Cd�
                                    Bx�%�V  �          @Ϯ�P  �����y����\CgJ=�P  �tz����
��\Ce��                                    Bx�&�  �          @�
=�U��(��l(��
��Cg{�U�z�H�z=q���Ce�{                                    Bx�&�  �          @�
=�[������n{�p�Ce���[��s�
�|���=qCd�                                    Bx�&"H  �          @�
=�g
=�u��qG��G�Cb���g
=�g
=�~�R���Ca                                      Bx�&0�  T          @�ff�b�\�q��u�p�Cb�H�b�\�c33������
Ca�                                    Bx�&?�  �          @�ff�mp��mp��mp���Ca��mp��`  �z=q�
=C_L�                                    Bx�&N:  �          @˅�j�H�z�H�Y���Q�Cb��j�H�n{�g��
��Cak�                                    Bx�&\�  �          @��
�n�R�x���Y��� �Cb:��n�R�l(��g
=�
G�C`��                                    Bx�&k�  �          @�(��c33�w
=�hQ��
Ccp��c33�i���vff�Q�Ca��                                    Bx�&z,  �          @��
�\���r�\�q��z�Cc�q�\���dz��\)��Cb�                                    Bx�&��  T          @���U��k��x���ffCcٚ�U��\����33�"{Cb                                      Bx�&�x  �          @�(��Q��mp������(�Cd�=�Q��^{��
=�%�Cb�f                                    Bx�&�  "          @˅�K��s33�\)�\)Cf��K��c�
��ff�%ffCdG�                                    Bx�&��  �          @ʏ\�L���k������33Cd�q�L���\(�����(�Cc�                                    Bx�&�j  T          @�=q�Vff�e��~�R�ffCb�{�Vff�U���%�C`�)                                    Bx�&�  �          @�33�C33�HQ������8�Ca���C33�7
=��\)�B  C_�                                    Bx�&�  �          @ə��'
=�dz�����5�Ci޸�'
=�S33��ff�@�\Cg�R                                    Bx�&�\  �          @ƸR�'
=�a���(��3�RCi���'
=�QG����\�>Q�Cgk�                                    Bx�&�  �          @�ff�G��y���W
=���CgL��G��l(��e����Ce��                                    Bx�'�  �          @�=q��(��\)>��@��C[E��(���Q�>k�@�C[k�                                    Bx�'N  �          @��
�����xQ�?��@�{CY�������y��>��
@8Q�CY�
                                    Bx�')�  �          @������H�s�
>��H@�ffCX�����H�u�>�=q@�CX��                                    Bx�'8�  �          @�Q����\�h�ý#�
��{CW�
���\�hQ쾀  �33CW��                                    Bx�'G@  �          @��H��33�p��>�{@C33CXaH��33�qG�=�?�ffCXxR                                    Bx�'U�  �          @ʏ\��z��j�H>���@,(�CW����z��k�=��
?8Q�CW�)                                    Bx�'d�  �          @��H����qG��:�H�ۅCZ}q����n{�s33�33CZ�                                    Bx�'s2  �          @����{�\)��
��33C]���{�vff�"�\��
=C\�H                                    Bx�'��  �          @�\)��z������   ���C^���z��x���/\)��{C]��                                    Bx�'�~  �          @�\)��p��}p��&ff��G�C]�H��p��s33�5����C\��                                    Bx�'�$  �          @�
=��33��Q��(Q����C^����33�u�7
=��  C]z�                                    Bx�'��  �          @�����H��������ffC^� ���H�w��*=q�ģ�C]��                                    Bx�'�p  �          @˅��z���=q�$z�����C`���z��z�H�3�
�ѮC_aH                                    Bx�'�  �          @�G����H�����"�\��ffC`�3���H�x���1G���G�C_��                                    Bx�'ټ  �          @�
=��G���Q���R��=qC`Ǯ��G��w
=�-p���33C_��                                    Bx�'�b  �          @�
=���\�|���!���(�C`!H���\�s33�0�����HC^�R                                    Bx�'�  �          @����
�xQ��\)���C_L����
�n{�-p���Q�C^#�                                    Bx�(�  �          @ƸR��Q��vff�Q���=qC^&f��Q��l���&ff��=qC]�                                    Bx�(T  �          @������\�i�������C\&f���\�`  �'
=��
=CZ��                                    Bx�("�  �          @��������u��\)����C]�������l(��p�����C\�
                                    Bx�(1�  �          @�G����
��Q��ff��G�C`5����
�y����\��{C_aH                                    Bx�(@F  �          @�=q�����|(���
=��ffC^u������u���z�����C]�                                    Bx�(N�  �          @\���H�qG��   ���C\�R���H�h���{��\)C\�                                    Bx�(]�  �          @�{��=q�j=q���R���
CZ����=q�b�\�������CY�                                     Bx�(l8  �          @ƸR���e�����
CY� ���^{�Q���=qCX��                                    Bx�(z�  �          @�G���=q�w
=����{C\5���=q�o\)������C[O\                                    Bx�(��  �          @Ǯ��=q�qG����H��z�C[�{��=q�i��������CZ�f                                    Bx�(�*  �          @�{��=q�c33�����z�CY����=q�Z�H�=q����CX�\                                    Bx�(��  �          @�{����^�R�p���
=CYǮ����U��*=q��p�CX��                                    Bx�(�v  �          @�z���=q�U�333����CY�3��=q�J�H�?\)���CXB�                                    Bx�(�  �          @�{����Z�H�(�����CXG�����R�\�Q���p�CW0�                                    Bx�(��  
Z          @�����G��Tz������CVǮ��G��L�������
=CU�=                                    Bx�(�h  T          @�����G��b�\�
�H��G�CY����G��Z=q�Q���CX��                                    Bx�(�  	e          @�z�����`�������33CY�)����XQ�����CX��                                    Bx�(��  
(          @��H����P  ����{CV������G�����G�CU                                    Bx�)Z  O          @�p����a녿���\)CY����Z=q�Q���p�CX
                                    Bx�)   �          @�  ��=q�^�R��
=���CW�{��=q�W
=�Q���\)CV�H                                    Bx�)*�  �          @����H�U���Q����CV����H�Mp��Q���
=CU��                                    Bx�)9L  "          @Ǯ��{�Y���޸R���CV�=��{�R�\��Q�����CU�                                    Bx�)G�  �          @�����p��P�׿޸R���CU����p��I����
=��ffCT�f                                    Bx�)V�  �          @�p������E���=q��(�CSu������=p�� ����  CR�                                    Bx�)e>  
�          @�����  �Dz����33CS����  �=p��G����CR��                                    Bx�)s�  �          @�(�����L(���Q���\)CUG�����Dz������CTG�                                    Bx�)��  
�          @�����G��L(������G�CU����G��C�
�����CT��                                    Bx�)�0  �          @������R�Mp��33���CV:����R�Dz���R��ffCU�                                    Bx�)��  T          @�z����R�J=q���\)CU�
���R�AG��!G���  CT��                                    Bx�)�|  
_          @��
��G��Dz��G���(�CT����G��;������ffCSz�                                    Bx�)�"  �          @��������I���{���CUJ=�����@��������CT&f                                    Bx�)��  
�          @��
�����J=q�ff��Q�CUaH�����B�\������CTO\                                    Bx�)�n  '          @�Q����R�A��	����{CT���R�9������Q�CS�H                                    Bx�)�  �          @�������s�
�G���33C^xR����p�׿�  �!G�C^{                                    Bx�)��  �          @����\����  �Y�����Cf���\����{����5�Cf�=                                    Bx�*`  
�          @���(���G���(��>ffC`c���(��}p������c\)C_�
                                    Bx�*  
�          @�  �n�R�}p���\)���Cb�3�n�R�u�ff��
=Ca�)                                    Bx�*#�  �          @�p��hQ���G��У���G�Cd��hQ��|(���{����CcW
                                    Bx�*2R  
�          @�Q��vff�}p���z�����Ca�=�vff�vff�����Q�Ca�                                    Bx�*@�            @��\�����mp�� ����p�C^�������e��{���C]��                                    Bx�*O�  "          @��\����`���z�����C\!H����XQ��G���\)C[)                                    Bx�*^D  T          @����z��@  ��G���
=CT����z��9����
=��
=CS�R                                    Bx�*l�  T          @�G������8�ÿ����Q�CS�
�����1G��33���CR�{                                    Bx�*{�  T          @��\�����7���\��  CS������0  �����p�CR�{                                    Bx�*�6  Y          @�=q����>�R�	������CU�\����6ff��
��CTk�                                    Bx�*��  �          @�  �����4z�������CS:������+��#33��33CQ�3                                    Bx�*��  
�          @�\)���
�/\)�%��  CR����
�%�/\)����CQG�                                    Bx�*�(  �          @�����33��=q�C33��{CH����33��z��I�����HCF�)                                    Bx�*��  �          @�G���  ���HQ���CK33��  ��{�O\)�CIY�                                    Bx�*�t  
�          @�������33�8����  CM�H��������AG�����CL                                      Bx�*�  O          @����Q�� ���1���=qCO����Q��ff�:=q���CNL�                                    Bx�*��  �          @�=q�����&ff�,����G�CP����������5��33CO&f                                    Bx�*�f  �          @�=q�����%��.{��33CPu��������7
=��
=CO                                      Bx�+  �          @����ff�%�*=q�ҸRCP޸��ff�(��333�޸RCOp�                                    Bx�+�  �          @�p����
�'
=�'���(�CQz����
�p��0����Q�CP\                                    Bx�++X  �          @�����p��.�R�'��θRCRL���p��%��1G���33CP�                                    Bx�+9�  T          @�Q������:=q�p���\)CT
�����1G��'��Ώ\CR�{                                    Bx�+H�  �          @�������0  �5���CS#�����%�>{��\CQ�f                                    Bx�+WJ  �          @�����\)�2�\�9����{CS�H��\)�'��B�\��
=CRY�                                    Bx�+e�  �          @�  ���%�E��ffCRO\���=q�N{�\)CP��                                    Bx�+t�  �          @�(��u��Z�H��G��\Q�C]��u��Vff��Q��}p�C]&f                                    Bx�+�<  �          @�����
�J=q�����CW�����
�B�\��\��{CVٚ                                    Bx�+��  
�          @��\��(��;���R�ə�CU����(��333�(�����CT��                                    Bx�+��  "          @�G���  �333�,����G�CUff��  �)���5���CS�R                                    Bx�+�.  T          @�{����HQ������  CY������@  �'
=����CXff                                    Bx�+��  �          @�
=�����O\)����ƣ�CZǮ�����G
=�#�
��CY�)                                    Bx�+�z  T          @�ff��=q�=q�9����\CO�{��=q�  �A����CNB�                                    Bx�+�   �          @�G���{�*�H�E��=qCS���{�   �N�R�=qCQc�                                    Bx�+��  �          @��R�����R�AG���CP�����z��H������COO\                                    Bx�+�l  �          @�����\)����@  ��ffCP5���\)�\)�G����CN�
                                    Bx�,  �          @\��
=��`���CM���
=��33�g����CK�                                    Bx�,�  �          @��
�������`  ��CM  ������
=�g
=��\CK
=                                    Bx�,$^  "          @��
�����[����CLY������33�b�\�ffCJs3                                    Bx�,3  
�          @�(���녿�{�p���p�CGz���녿��u�33CE@                                     Bx�,A�  �          @�z��������q����CC�������{�u��CAk�                                    Bx�,PP  �          @�p���p���(��u�33CB�f��p����\�y�����C@W
                                    Bx�,^�  "          @�(����\��  �w���
CC:����\��ff�{���C@�H                                    Bx�,m�  �          @�G���p�����x��� ffCDٚ��p�����|���#�\CBp�                                    Bx�,|B  "          @�z���{�}p��\)�+ffCAQ���{�J=q��G��-��C>�f                                    Bx�,��  T          @�  �|(�������p��:33C8�q�|(������:�HC5�                                    Bx�,��  
�          @��
���Ϳ����x���&
=CGT{���Ϳ�G��}p��)��CD�)                                    Bx�,�4  
�          @�z��w
=�\(���33�?  C@���w
=�#�
��z��A33C=k�                                    Bx�,��  T          @�z��i���fff�����I33CA�
�i���+���=q�K��C>p�                                    