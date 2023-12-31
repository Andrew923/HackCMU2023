CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230510000000_e20230510235959_p20230511021735_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-11T02:17:35.853Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-10T00:00:00.000Z   time_coverage_end         2023-05-10T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�/R@  
f          AL��A���
=���\��{C���A����=q��p���\)C��                                    Bx�/`�  H          A$  @����(��*�H�w\)C���@��������  �33C�Y�                                    Bx�/o�  �          Ac
=A#��'
=@�ffB�
C���A#���  @���Aϙ�C��)                                    Bx�/~2  �          A�33A\)@ə�AO
=BI(�B�A\)>�  Af=qBk=q?�
=                                    Bx�/��  
�          A��A&�\@�33AH  B>�A���A&�\���ATQ�BN�C�                                      Bx�/�~  �          Aq�@���A(�A$��B8�HBz�\@���@g
=APz�B�B(�                                    Bx�/�$  |          A�  @��@��RArffBm�\BQ�@������A��B�\C�
                                    Bx�/��  
�          A�z�@�
=A�HAp��BY��Bt  @�
=@{A�
=B�Q�A��
                                    Bx�/�p  
�          A��@���A/�
AV�\B@p�B���@���@��\A�Q�B�p�B&                                      Bx�/�  
�          A��?�{AQ�AE��B,�B��?�{@�p�A���B�Q�B��
                                    Bx�/�  �          At�����
AE녾�\)��{B����
A5��@�{A��\B���                                    Bx�/�b  �          Ar�R��A`zῪ=q����B��H��AT��@�A���B�                                    Bx�0  �          A~�\��{A]p��������B�\��{AW\)@w
=AaG�B��H                                    Bx�0�  �          A}��� (�AV�R�-p��p�B��� (�AU�@J=qA8(�B�
=                                    Bx�0T  �          A�p��
{AW33�"�\���B�ff�
{ATz�@U�A=B�\                                    Bx�0-�  �          A������AU���P���;�B�R���AW�
@(Q�A�RB�33                                    Bx�0<�  T          Aqp���A5��33����B�#���AJ�\���ÿ��\B�\)                                    Bx�0KF  T          Ag�
�G�A/�������RB���G�A:�H?fff@h��B�aH                                    Bx�0Y�  �          Ao���A;�
�K��EB����A@  @G�@��\B��                                    Bx�0h�  �          AuG��  A<z��(Q���C !H�  A<��@!�A��C {                                    Bx�0w8  �          Ayp��p�AF�H�S33�C�B��)�p�AJ�R@�A ��B���                                    Bx�0��  T          A{33���AL���p����B������AH��@U�AD  B�                                    Bx�0��  �          Au��
=A;\)����p�CT{�
=A-�@���A���Cs3                                    Bx�0�*  �          Av=q�'
=A3�>�z�?��C�f�'
=A!p�@�{A�ffC�R                                    Bx�0��  �          Al���{A9�>�33?���C (��{A&�R@��A�\)C:�                                    Bx�0�v  �          Av�R�p�AO��+���RB�k��p�AAG�@�G�A��\B�33                                    Bx�0�  �          Ak\)��ffAE���\�}p�B�z���ffA:�R@��RA��HB��=                                    Bx�0��  �          Afff� ��A?33��ff���B��H� ��A0��@���A�G�B�\                                    Bx�0�h  �          Am�ffAC�
��\)��B�Q��ffA>�\@XQ�AR�HB�Ǯ                                    Bx�0�  
�          An{���AH  ��Q�����B��3���A?�@w�AqB���                                    Bx�1	�  �          At  �
=AJ=q��(���33B�u��
=A@(�@�(�A{
=B�33                                    Bx�1Z  T          AuG����\AR�\>��?�B�z����\A>�R@��HA�33B���                                    Bx�1'   T          Av=q��Q�AZ�R?@  @1�B�Q���Q�AB{@ʏ\A\B�\                                    Bx�15�  h          A{�
��z�A^ff?���@�33B�R��z�A@��@�=qA��
B�z�                                    Bx�1DL  h          A|z�����Ap�AEp�BR�RB�k�����@G�AiG�B���C��                                    Bx�1R�  h          A����ڏ\A�AJ=qB<G�B����ڏ\@`  Aw�B~p�C�
                                    Bx�1a�  �          A��R��  AP(�@��A�ffB���  A��A��B{C .                                    Bx�1p>  T          A�(��\)Af�H��\��p�B�3�\)AU@�\)A�z�B��q                                    Bx�1~�  �          A�=q�(�A_�?�  @��B�#��(�A@Q�@�A�z�B�.                                    Bx�1��  T          A�����G�AXz�@�33A��
B��)��G�A z�A*�RB (�B��                                    Bx�1�0  �          A�����HAV�R@�A��B�B���HAffA+
=BCk�                                    Bx�1��  �          A���  AQ��@E�A*�\B�Q��  A*�HA
=A�{Cff                                    Bx�1�|  T          A�\)�%G�A@Q�@u�AYG�C���%G�A=qA�HA���C	�R                                    Bx�1�"  T          A��z�AB�\@�AtQ�B��R�z�A{A��BC��                                    Bx�1��  T          Az�R��{Ac�@g
=AU�B��f��{A7�
A�B�B�3                                    Bx�1�n  �          Au��p�A\��@u�Ag�B��)��p�A0Q�A=qB�B�W
                                    Bx�1�  �          A�p��{�AUA�A�  B�Ǯ�{�AAM�BT{B���                                    Bx�2�  T          A������Aj�R@X��ADQ�B������A@  Ap�B
�HB�q                                    Bx�2`  h          A���z�HAk
=@�Q�A���B��
�z�HA-A;\)B5�RBۮ                                    Bx�2   �          A�Q��|��Ar�H@��\A�
=B�#��|��A>{A)B!{Bؽq                                    Bx�2.�  �          A�ff�b�\AB=qA)�B�\BԊ=�b�\@���Ai��Byp�B�33                                    Bx�2=R  �          A{�
�i��A>ffAp�Bp�B���i��@޸RAYBo�B�aH                                    Bx�2K�  �          A~�H�^{A*=qA4(�B4��B���^{@�z�AiB�B�
=                                    Bx�2Z�  �          Aw��z�AQ�A;\)BHG�B�k��z�@|(�Ai�B�(�B��                                    Bx�2iD  �          Ap  � ��A��A;�BN�B��� ��@c33Af�\B�ffB��                                    Bx�2w�  ,          Au��Q�A{AN=qBd�HB�z��Q�@(�AqG�B�.B�(�                                    Bx�2��  
�          Au����
=@���AW�
B{�B�Ǯ��
=?8Q�AqG�B�� C�
                                    Bx�2�6  �          AY��.{@߮A6{BiG�B�
=�.{?��
AS�B���Bݔ{                                    Bx�2��  
�          A`��?�=qA��A4��BX�\B�?�=q@5�AY�B��fB�\                                    Bx�2��  
�          AU?�z�A (�A33BF�RB�=q?�z�@Y��AAB�Q�Br��                                    Bx�2�(  r          A�  ���@љ�Af�HB�33B�
=���>#�
A}p�B��HC-u�                                    Bx�2��  "          A���@�
=���Ab{BmffC��
@�
=�-p�A*�\B!�HC��                                    Bx�2�t  �          A��
@,��?�Am��B�Q�B33@,�����HAd��B��RC��{                                    Bx�2�  
�          A����K�A
�HAl(�BgB�=q�K�@   A�  B��C�f                                    Bx�2��  >          A�33�<��@�
=A�ffB�aHB�R�<�Ϳ�  A���B��HCR��                                    Bx�3
f  "          A����Q�A=qAz{B^��B����Q�@1G�A�Q�B�(�CQ�                                    Bx�3  �          A�z��Dz�A�A�\)Bi�B�aH�Dz�@p�A�33B�aHC=q                                    Bx�3'�  
�          A������A"�RA{�
B\Q�B�ff����@@  A��B�\C+�                                    Bx�36X  �          A�33��  A'33Ax��BY  B��)��  @UA�p�B�p�C�                                    Bx�3D�  �          A�ff��  A0��Aw�BRQ�B����  @{�A��HB�p�C	L�                                    Bx�3S�  T          A�Q���\)A=qA���Bf��B�\)��\)@�A��RB�C��                                    Bx�3bJ  "          A����Q�AA�Ap  BHG�B�Q��Q�@��A��RB���B�u�                                    Bx�3p�  
�          A�z����A   AzffBQ=qB�z����@:=qA��RB�
=C��                                    Bx�3�  T          A��R����A"=qAz�HBQ(�B��=����@B�\A�p�B��3Cs3                                    Bx�3�<  �          A�z��ҏ\A)G�Ap��BIz�B�R�ҏ\@mp�A�{B��C�
                                    Bx�3��  �          A����ᙚA(Q�An=qBFz�B��3�ᙚ@n{A��RB��)C5�                                   Bx�3��  @          A���G�A<��AK33B%=qB�\)�G�@���A���Bh��C8R                                   Bx�3�.  
�          A�\)��AJ{A��BG�B�G���@�G�A\z�BQC�                                    Bx�3��            A����p�A�HAH  BB�\B���p�@l��As\)B�
=C�f                                    Bx�3�z            A������A$(�A;�B4
=B�Ǯ���@���AmG�B}�HC��                                    Bx�3�   �          A�����{AJ�HA5�B
=Bޞ���{@��Av�RBrG�B�G�                                    Bx�3��  |          A�Q��AG�Ahz�A)p�B  B˅�AG�AffAw�BhQ�B؏\                                    Bx�4l  �          A�����\)Ak�A�\A�Q�B�8R��\)A\)Ak
=BPB�G�                                    Bx�4  T          A����{A]�A��A�{B� �{A�A\z�B@(�C��                                    Bx�4 �  
�          A��
�z�AZ�HA��A�Q�B��
�z�A33AT  B7(�Cu�                                    Bx�4/^  
Z          A����=qAi�A
�HA݅B�k���=qA�A[\)B={C 
                                    Bx�4>  
�          A�\)��\AhQ�A  A��HB����\A!��AT��B6��C �                                    Bx�4L�  	�          A�����HAf�R@߮A�{B�{��HA((�AAp�B�C\)                                    Bx�4[P  
�          A�(�� ��An�H@���A\)B���� ��A;\)A)��B	��C��                                    Bx�4i�  
�          A����0z�Af�H@z�HAB�HB��0z�A;�
AQ�A�\)C8R                                    Bx�4x�            A�(��3\)Ae�@��HAV=qB���3\)A8Q�A�A�33C8R                                    Bx�4�B  
�          A�  �8(�AP  @��HA`��C���8(�A$(�AG�A��C
G�                                    Bx�4��  
�          A�G���AN{@��HA���B�����A�A-�B"�HC�                                    Bx�4��  
(          A���\)AK\)@��A£�B�B��\)A(�A5�B*  C33                                    Bx�4�4  
�          A�  ��
AK\)@�G�A�B����
A��A(Q�B=qC޸                                    Bx�4��  
(          A����33AA��@��A��RB�u��33A	p�A,  B"{C�3                                    Bx�4Ѐ  
(          A�  �љ�A9��AA��RB���љ�@���A?�
BH�HC
=                                    Bx�4�&  
�          A�p��-G�Ab�R@\(�A.�\B����-G�A;�A
�HA�=qC�                                     Bx�4��  
(          A����*�\Ae�@[�A-�B�Q��*�\A=A�A�G�C�                                    Bx�4�r  
�          A��&{AJ�H@�p�A�33CO\�&{A��A0��B�
C
                                    Bx�5  
�          A��\�(��AAp�@�p�A�z�C!H�(��A�A4��B�
C�
                                    Bx�5�  
�          A���<(�Ac�@G�@�Q�C�\�<(�AC�
@�33A��HC�
                                    Bx�5(d  H          A�
=�0(�Amp�@\��A)p�B�#��0(�AE��A�RA�\C��                                    Bx�57
  
�          A�p��$  A��@N�RA��B���$  A[
=A�A�RB���                                    Bx�5E�  
�          A�ff�)��A��@(Q�@�  B�ff�)��A^�HA
�RA��HB��=                                    Bx�5TV  
�          A�\)�*{A��?E�@G�B���*{Aj�R@�ffA��B��f                                    Bx�5b�  "          A���*�RA��?�=q@�z�B�Ǯ�*�RAep�@�A�  B�L�                                    Bx�5q�  
�          A�ff�*�HAvff?�
=@fffB�z��*�HA\Q�@�  A��B���                                    Bx�5�H  
�          A����)p�Ax(�?(��@G�B����)p�Aa�@��A�p�B��f                                    Bx�5��  
�          A����#�Aw�?��@~�RB����#�A\��@��
A�{B�#�                                    Bx�5��  
�          A�33��\Ay�?�33@e�B�z���\A_�
@�G�A��
B�B�                                    Bx�5�:  �          A��#�Ay���p�����B��=�#�Ao33@��\Ab�HB��                                    Bx�5��  	�          A�(����A�{�����33B�\)���A}��@��
AjffB�p�                                    Bx�5Ɇ  �          A�z��(  A�33�����L  B�=q�(  A��R@�@�Q�B��)                                    Bx�5�,  
R          A���-�Az�\��{�h��B�G��-�A��H?��@�=qB���                                    Bx�5��  H          A��H�+\)Am����
���B��3�+\)A��H�����B���                                    Bx�5�x  �          A���1�Ag���{��{B�{�1�A��ÿ�z��X��B�33                                    Bx�6  
�          A�
=�!G�Ac
=����  B�Ǯ�!G�A�=q�:�H�	�B�q                                    Bx�6�  �          A�{��
A9���;
=�\)B���
Ar�\����p�B��                                    Bx�6!j  
(          A�G��ҏ\@�
=�X  �R=qC���ҏ\AG�
�����B�=                                    Bx�60  �          A~=q��33A�\�G��N��B� ��33AG�
����z�B�k�                                    Bx�6>�  �          At����
=A%���
��RB��f��
=AR�H��(����B�                                    Bx�6M\  �          Ac
=�ָRA:�R@��A��B�Ǯ�ָRA  A(�BffB��                                    Bx�6\  T          A�\)��RA\��������p�B�Ǯ��RAs��\(��333B�3                                    Bx�6j�  "          A�33���AQp���Q���  B��R���Ad�;�
=��B�                                    Bx�6yN  `          A�  �=p�AS\)?�\)@���C�H�=p�A8��@�{A�=qC�q                                    Bx�6��  �          A�p��&=qAY��1G��\)B��)�&=qAY�@ ��A��B���                                    Bx�6��  �          A��R�%�Ar{�hQ��2�HB��)�%�Av=q@33@ᙚB���                                    Bx�6�@  �          A����,��Ax���{��{B��=�,��Au��@aG�A)B�=q                                    Bx�6��  `          A�(��-�Al����\���HB��\�-�Aip�@Y��A)p�B�aH                                    Bx�6  �          A�  �D��A[33�����G�C��D��AT��@j�HA7\)C�=                                    Bx�6�2  �          A���A�AX��>�{?���C�=�A�AG\)@��A�
=C5�                                    Bx�6��  �          A�(��C
=AS�?�33@�{C���C
=A<(�@ƸRA�
=C�                                    Bx�6�~  �          A��H�M��AE�@�Q�AJ�HC��M��A   A��A�ffC�                                    Bx�6�$  �          A���;�
AF�H@�  A�z�C\)�;�
A��A�B 
=C�R                                    Bx�7�  `          A�{�:ffA:�H@g�AEC��:ffAQ�@�A�ffC                                    Bx�7p  �          A�Q��%�AD�������\)C���%�Alz��w��A�B��
                                    Bx�7)  �          A�{�9AO\)��G���G�Cٚ�9A`  �\)��(�C��                                    Bx�77�  �          A����7
=AD  �Q���=qC��7
=Af�H�K����C ff                                    Bx�7Fb  �          A�{�6�HAB=q�	G����
CG��6�HAg33�aG��,��C Y�                                    Bx�7U  T          A����P  AV=q�y���<��C(��P  A^=q?��@w�C)                                    Bx�7c�  
�          A��\�:ffA<(��Dz��'�
C���:ffAA�?��H@�\)C��                                    Bx�7rT  H          @�녾u����?�B�(�Cn��u�L��?p��BA(�C}
=                                    Bx�7��            A*=q?�G����A�RBa{C���?�G���@��BffC�Ff                                    Bx�7��  
�          Ap����
���
@�(�BB�HC~�=���
����@�(�A��HC�Z�                                    Bx�7�F  
&          A(  ?+���33A	G�Bb�RC�k�?+��p�@�ffB��C�L�                                    Bx�7��  �          A[��(�����RAFffB�(�C����(�����A��B3G�C�
=                                    Bx�7��  h          AHQ�@#�
���\A  Bq�RC���@#�
��(�@޸RB#Q�C�J=                                    Bx�7�8  �          AC
=@�  ��{@�Q�B��C�o\@�  ��\@���A��C��                                    Bx�7��  �          AB�\@�p��ᙚA ��B&��C�e@�p���
@�ffA��HC�l�                                    Bx�7�  �          A=��@�=q��z�@У�BG�C�C�@�=q�  @Y��A��C�,�                                    Bx�7�*  T          A:=q@�=q�O\)A&=qB}�C��q@�=q�أ�AQ�B<��C�~�                                    Bx�8�  �          A,(�@k��VffA\)Bx��C���@k���G�@�B5�C��                                    Bx�8v  �          A\)@i���%AB|�C�W
@i����33@��HB>��C���                                    Bx�8"  T          A�H@��
��(�@�p�BU�RC���@��
��ff@���B-�\C���                                    Bx�80�  |          A#
=@(���\A�B�p�C�@(���33A�
BY��C�=q                                    Bx�8?h  T          AK�?�p����AG�
B�aHC��{?�p���A333BxG�C�}q                                    Bx�8N            AC�
?h��?��
A@  B��BBG�?h���U�A9G�B�ǮC��f                                    Bx�8\�  �          AS�@S33���A6{B��C�.@S33�(�AQ�B5�C��                                    Bx�8kZ  
(          AU@Y����A>�RB}=qC�O\@Y���A{B2��C���                                    Bx�8z   T          AYp�@N�R�a�AJ�HB���C�8R@N�R��=qA)p�BM��C�8R                                    Bx�8��  �          AZ=q@7��J=qAO\)B��\C�%@7����A/�
BV��C�ff                                    Bx�8�L  T          A[33@(���@��AQ��B�  C���@(����RA2�HBZC��q                                    Bx�8��  T          Ah  @9���P��A]G�B���C��R@9����ffA<z�BY33C��                                    Bx�8��  
�          AW33@��+�AP  B��RC�7
@���33A3\)Bap�C��                                    Bx�8�>  T          AT��@�
���AN�\B��\C�Y�@�
��G�A3�Bf��C�j=                                    Bx�8��  "          AW33?Ǯ��RAQ�B���C�#�?Ǯ��{A6ffBh�C�Y�                                    Bx�8��  �          A\(�?�����AW�
B���C�L�?�����
A=G�Bm��C���                                    Bx�8�0  �          AS�?�Q��G�AO�B���C�˅?�Q���{A5��BlffC��                                    Bx�8��  T          AYp�?u�33AV=qB���C���?u���
A=�Br  C�!H                                    Bx�9|  �          AY@���RAS\)B��qC��q@���ffA8  Bg=qC�                                      Bx�9"  
�          AUp�?�=q� ��AO�B�{C�f?�=q��z�A4z�Bf�
C�l�                                    Bx�9)�  T          AP  ?:�H��Q�AM��B��3C���?:�H��=qA733Bw��C�o\                                    Bx�98n  
�          AY�?��
���AW�B���C��?��
��\)A@��Byp�C��3                                    Bx�9G            AZ=q?333��\)AX(�B�#�C��f?333��
=AA��Bz��C�33                                    Bx�9U�  
Z          AW�>�33���AUB�C�
>�33��{A@��B~��C���                                    Bx�9d`  
�          AS\)���fffAR�\B��C�:�����A@z�B���C�^�                                    Bx�9s  �          AU��z��AT��B��=C[�H�z�����AD��B��C��)                                    Bx�9��  .          AU녿&ff�#�
AUB�k�C533�&ff��(�AHz�B�{C�                                      Bx�9�R  \          AU�>�R?�Q�AM��B�#�Cu��>�R�6ffAJ=qB��HC_��                                    Bx�9��  �          AW
=�(Q�?�AP��B�� Cff�(Q��2�\AM�B��
Cb��                                    Bx�9��  
�          ATz���  �#�
AAG�B�\C4���  ��{A5G�Bs�C]p�                                    Bx�9�D  �          AXz��ָR>��HA;\)Bpz�C/���ָR�c�
A2�RB_�CO�                                    Bx�9��  h          A\������@'�A=�BkG�C}q���Ϳ˅A@(�BqQ�CAff                                    Bx�9ِ  @          AeG��ۅ@9��A=��BgQ�C��ۅ����AB{Bp(�C>�f                                    Bx�9�6  T          Ao�
���RA
=A0Q�B=��B�����R@|(�ARffBu��C�\                                    Bx�9��  
�          Ajff��ffAp�A*=qB9�B�����ff@�
=AMp�BtffC}q                                    Bx�:�  �          A`z����\@��HA0��BR�B��q���\@:=qAL��B�\C��                                    Bx�:(  
�          AdQ���=qA
=A
=B��B�#���=q@���A0��BL33C��                                    Bx�:"�  h          Aj�H�r�\A�A/�BBB�33�r�\@�(�AUp�B��3CG�                                    Bx�:1t  
�          Ap����
=A�RA�B$33B�Ǯ��
=@�  AJ=qBe�CaH                                    Bx�:@  "          Aw\)�{A:�\@�z�A��B��\�{A�A��B=qC��                                    Bx�:N�  
�          Ar�R�5��A�R������(�CL��5��A�@
�HA
=C�H                                    Bx�:]f  |          A�G���
=@_\)A	�BfQ�Ck���
=>\A��B���C.�=                                    Bx�:l  T          A�(�?8Q����A�33B��fC��?8Q�� ��Aj{B]�RC�\                                    Bx�:z�  �          A���{���
A��RB�ffCY8R��{���
AxQ�B��C�/\                                    Bx�:�X  �          A���aG���A�=qB�  Cv�׿aG�����Ar=qB}��C��                                    Bx�:��  �          A�(���녿��A��B��fCy@ ����У�Ar{B�Q�C�0�                                    Bx�:��  "          A�=q�n{��A�33B��qCp�R�n{��G�AnffB�\C�AH                                    Bx�:�J  
�          A�=q�E���RA��\B�33C|�׿E���33Al��Bw��C�/\                                    Bx�:��  `          A���Tz��+�A�33B��{C|� �Tz���
=AiG�Bt��C�                                    Bx�:Җ  
�          A���Tz��1G�A~{B�aHC}Q�Tz����
A`��Brz�C���                                    Bx�:�<  �          A�=q�@  �aG�A}B�k�C���@  �	�A\��Bh=qC���                                    Bx�:��  
�          A~�\��
��\)Ay�B��=CW���
��{Ac\)B~(�Cz��                                    Bx�:��  
�          A}��>�R�n{Axz�B�B�CET{�>�R���Ae��B���Cs��                                    Bx�;.  "          A���(���(�A}p�B���CS#��(���33Ag�B�(�Cy�q                                    Bx�;�  �          A����p��{A��HB�\Cg:��p���ffAj=qBu�C��                                    Bx�;*z  �          A�{��ff�L��A��RB�Q�Cp����ff���Ab�\Bl=qC��H                                    Bx�;9   �          A����ff�B�\A�ffB�.CoJ=��ff��Ab�HBnffC��q                                    Bx�;G�  "          A�(��,����A�
=B�{CZ���,������AlQ�Bw��Cy��                                    Bx�;Vl  �          A��\�I���
=A���B��CUٚ�I����  Ah��Bu��Cv��                                    Bx�;e  T          A��\�   �#�
A�B�Cg���   ��Adz�Bt�\C@                                     Bx�;s�  �          Au��(Q��{Ak�
B�u�C_+��(Q����
AR=qBo��Cy�3                                    Bx�;�^  T          AZ{�Dz���AS�
B�k�C?&f�Dz����AEG�B�Cm��                                    Bx�;�  "          Ah(���ff�8Q�Ab{B���Cu�f��ff��G�AG
=Bm  C��                                    Bx�;��  �          A[\)�y�����APz�B�� C:�{�y������AC
=B~33Cf                                    Bx�;�P  �          AiG��g��{A^ffB�
=CS���g��ҏ\AG33Bk��Cq.                                    Bx�;��  �          Anff������Ac�B�aHCD�R�����G�AP��BtCi�\                                    Bx�;˜  �          As33���\?L��A`(�B�#�C+�f���\�o\)AXQ�B~Q�CU��                                    Bx�;�B  r          As
=�=q@޸RAffBffC޸�=q@j�HA.=qB7�C��                                    Bx�;��  �          Am��A��@�A�p�C�q�@��AffB!�RC�{                                    Bx�;��  
�          Amp��(�@�z�@���B �C��(�@�
=A   B)��C�                                    Bx�<4  z          Ajff�=q@޸RAQ�B	p�C���=q@~{A ��B-G�C(�                                    Bx�<�  T          Ac��@��A��B#z�Cp��@�A((�B>�\C'z�                                    Bx�<#�  T          AYp��	�@��HAG�B-ffC���	�?��A&=qBGG�C):�                                    Bx�<2&  �          Abff�33@�G�A�B"��C�f�33?�\)A%��B<33C(��                                    Bx�<@�  T          Ahz��33@��
A��B*�
C�3�33?��
A-��BB\)C*�=                                    Bx�<Or  �          A]p��
=q@�\)A�B2z�CG��
=q?�
=A+�BK  C*�{                                    Bx�<^  "          A_
=�p�@�(�AG�Bp�C��p�@\(�A�\B633CE                                    Bx�<l�  "          AVff���@���A�HB(33C� ���?�  AffB?�
C*J=                                    Bx�<{d  h          AS�
�	�@���AffB,�RC���	�?���A�BAz�C,�f                                    Bx�<�
  �          AL����\@��
@�33B�C
��\@Mp�A��B4��C#�                                    Bx�<��  �          AY���\@���@�A���C^���\@��A ��B��Ck�                                    Bx�<�V  "          AP(��33@���@�p�A�Q�C
��33@�\)@�\B��CaH                                    Bx�<��  
�          AP(���\A=q@s�
A��\C	���\@�G�@�\)A��Cp�                                    Bx�<Ģ  �          AK��z�A��@Z=qAx��C	p��z�@ڏ\@��A�\)C�f                                    Bx�<�H  �          AAG���p�A
=@{�A�z�C
=��p�@�=q@ȣ�A��HCO\                                    Bx�<��  �          AB=q�ff@��R?�@��RC�H�ff@�=q@~{A�\)CL�                                    Bx�<�  �          AD(��{@�z�@33A=qC+��{@�(�@�(�A���C�                                    Bx�<�:  T          AS\)��A�\�����Q�C	+���A�?ٙ�@��C	z�                                    Bx�=�  T          AV=q�=qA  ��
��HCn�=qA(�?!G�@.{C�H                                    Bx�=�  �          A[33�{A(��B�\�Mp�C)�{A  ��\)��z�C��                                    Bx�=+,  "          A[�
�
=A�H�HQ��S\)C���
=A33�B�\�E�C��                                    Bx�=9�  "          AR�\�ffA
=���Ϳ�G�Cn�ffA
=q@A&=qC	k�                                    Bx�=Hx  T          AH������A��@��A���C������@���@��
BQ�Cn                                    Bx�=W  
f          A@Q���Q�@�
=@�  B��C����Q�@^�R@�p�B+�C�
                                    Bx�=e�  H          AH  ����@�{A
=B+��C�f����?�A\)BG
=C&!H                                    Bx�=tj  �          A?�
���@�ff@��HA���C)���@��@�z�B�C^�                                    Bx�=�  
�          AM����@�\@���A���CG���@���@�=qA�C��                                    Bx�=��  �          AZ=q�'
=@���@�\)A�(�Cz��'
=@��@�(�B{C�\                                    Bx�=�\  T          AUp��#\)@���@�G�A�(�C��#\)@�33@�=qA�Q�C33                                    Bx�=�  "          AR=q�ff@���@`��Ax  Ch��ff@Ϯ@��RA�(�C�                                     Bx�=��  
�          AV=q�A
{@/\)A<��C
�=�@�@�{A�Q�C�                                    Bx�=�N  T          A\����AG�@=qA!G�C�R��A�H@���A��
C
�H                                    Bx�=��  �          A_�
� z�A��@�33A��C\)� z�@�p�@��HA��C^�                                    Bx�=�  �          A]G��$  @�@���A��C�$  @�ff@�p�B�
C{                                    Bx�=�@  T          A]G��&{@�ff@�p�A���C+��&{@�z�@��RB��C�                                     Bx�>�  "          AZ�H�.�R@��
@��A���C��.�R@�33@�
=A���C�                                    Bx�>�  "          AYG��,��@Ǯ@��A�\)C�q�,��@��
@�  A��C��                                    Bx�>$2  T          AX���,Q�@���@��HA�C�\�,Q�@`  @�ffB\)C!�q                                    Bx�>2�  �          AR�\�#�@��R@���A���CǮ�#�@�  @�G�BG�C��                                    Bx�>A~  
(          A\���'\)@θR@�Q�A��
CL��'\)@�33@��
B33Ck�                                    Bx�>P$  T          A\���'�@���@�A�p�C���'�@�Q�@��A��RCW
                                    Bx�>^�  T          AV�R�)��@��@�
=A��C8R�)��@���@ڏ\A�G�C��                                    Bx�>mp  �          AI��� z�@��@���A��Cu�� z�@��\@ÅA��
Cs3                                    Bx�>|  �          AN�R�%�@Ǯ@�p�A�G�C�{�%�@�(�@���A�(�C޸                                    Bx�>��  �          Aj�R�=�@��H@�Q�Aۙ�C�=�=�@J=qA (�Bz�C%�                                    Bx�>�b  �          Ag��=��@��
@��HA�C���=��@Tz�@��
A�  C$W
                                    Bx�>�  "          Ae��>ff@�p�@�
=A�=qC���>ff@^�R@��A�33C#��                                    Bx�>��  T          Afff�;�@�Q�@�G�A��C�
�;�@^{@�33A���C#�                                     Bx�>�T  "          Ae���8Q�@�Q�@�ffA��Cu��8Q�@G�@�BQ�C$�{                                    Bx�>��  
�          Ah  �6=q@�z�@�A�{C�3�6=q@5�Az�B��C&�                                    Bx�>�  T          Ag��5@�33@�  A�  C�)�5@!G�A	�Bz�C'}q                                    Bx�>�F  
�          Ah���0z�@*=qAB  C&s3�0z�#�
A�
B"��C40�                                    Bx�>��  
�          Al  �3�@'
=A�HBG�C&��3�����A��B!�\C4�                                    Bx�?�  �          Al���;�
@0��A��BG�C&Ǯ�;�
>k�A�B��C2޸                                    Bx�?8  �          Am���5p�?�z�A�B�RC*s3�5p��O\)Az�B�HC8
                                    Bx�?+�  "          An�H�)p�?fffA'�B2\)C/��)p���A$��B.C?�                                    Bx�?:�  
�          Al�����>\A1p�BB33C1Ǯ����0  A,(�B:ffCC�                                    Bx�?I*  �          Af�R��=uA.=qBC�C3�����?\)A'�B:
=CE�                                    Bx�?W�  "          A^{�p��\)A'�BD{C4���p��C�
A (�B8�HCF�
                                    Bx�?fv  
�          A`(��!G�?�{A��B-\)C,O\�!G���\)Ap�B-\)C;�R                                    Bx�?u  |          Ac\)�p��
=A.=qBHG�C7���p��eA$��B9�HCI��                                    Bx�?��  �          Ac���H�aG�A0Q�BKffC9����H�y��A%��B:�HCK��                                    Bx�?�h  
�          Ad���{��A,z�BC�C6�{�{�Z�HA#�B6�RCH�                                    Bx�?�  	�          Ag��*{?z�HA  B)�C.���*{�޸RAffB'ffC=O\                                    Bx�?��  T          Aep��0Q�?���A33B�\C)�R�0Q�
=A=qB�C7�                                    Bx�?�Z  �          Ab�R��?@  A (�B4
=C/�����G�Ap�B0  C?u�                                    Bx�?�   "          Aa��%��?�=qAz�B)�HC.
=�%������A\)B(ffC<�)                                    Bx�?ۦ  �          Ad���,��@z�A=qB��C)#��,�Ϳ   AB#�C6��                                    Bx�?�L  �          Ac��0Q�@%A	��B�
C&���0Q�>��A�B��C3:�                                    Bx�?��  `          Ae���4��@@��ABQ�C%��4��?!G�AB\)C0�{                                    Bx�@�  �          AeG��6�\@P��A Q�BG�C$
=�6�\?xQ�A	BC/(�                                    Bx�@>  �          A`z��1�@h��@�G�B��C!�)�1�?�33A�Bz�C,�\                                    Bx�@$�  T          AZ�\�,  @w
=@�ffBp�C B��,  ?��HA\)B�C*��                                    Bx�@3�  �          AI��z�?�p�@��HB�C(�{�z�\)@��HBC4��                                    Bx�@B0  
�          A/�
�	��@z�@У�B�C&z��	��>�  @��HB�
C2^�                                    Bx�@P�  �          A=�1�?Tz�@�=qA�{C/�3�1���33@�z�A���C5�{                                    Bx�@_|  "          A;
=�0��?�(�@HQ�Ay�C)�f�0��?��@b�\A��
C.�                                    Bx�@n"  �          A733�   @hQ�@�  A���C ��   @�@�=qA��HC&Q�                                    Bx�@|�  �          A>ff�33@�  @���A�z�C���33@e@��A�\C�                                    Bx�@�n  �          A;
=�G�@�G�@��HA�(�C�)�G�@e�@�  A��
C                                      Bx�@�  T          A:�H��@�  @�
=A�{C����@!G�@�(�B�C$L�                                    Bx�@��  `          A3����
?���A ��B>ffC%����
��p�A  BD�\C6�                                    Bx�@�`  z          A2�H��p��$z�A�Bj�\CNk���p����RA\)BF��C_ٚ                                    Bx�@�  T          A"�H���H?+�@���B)�RC.+����H�p��@���B(z�C<
                                    Bx�@Ԭ  	�          A��p�?Ǯ?޸RA9C)c��p�?�ff@A_�C,�\                                    Bx�@�R  
�          A
{�	��=��
�.{��{C3}q�	��=��\)�p��C35�                                    Bx�@��  
�          A�������=���?(��C;T{����=q�B�\��G�C;@                                     Bx�A �  	�          A��=q?W
==L��>��RC.G��=q?L��>��?��
C.�\                                    Bx�AD  
�          A����R@��>�Q�@�HC#Q���R@G�?z�H@�33C$u�                                    Bx�A�  
�          A����Q쾅�@�z�A�\)C6��Q쿧�@{�A�z�C>B�                                    Bx�A,�  	          A���z��Q�@�
=BC=qCM����z��y��@���B#{CZ�{                                    Bx�A;6  
�          AG���  � ��@�B?=qCO�H��  �|��@��HB�
C[�q                                    Bx�AI�  	�          @���z���\@��BLCO����z��r�\@��
B+(�C\                                    Bx�AX�  
�          A���=q�C33@ȣ�BK�CX�\��=q����@�G�B#�Cd(�                                    Bx�Ag(  �          A{����?�Q�@�(�B3�C(^�������{@��B7p�C7\)                                    Bx�Au�  	�          A\)���H>#�
@�Q�B2\)C2p����H���@��B,��C@��                                    Bx�A�t  
4          A����ÿc�
@��B&\)C<h�������R@�Q�B��CHO\                                    Bx�A�  
�          A�
��=q�{@���B4
=CJ���=q�g
=@���B��CV#�                                    Bx�A��  
�          A\)���H��@��B+p�CJ+����H�g�@�33B=qCT�H                                    Bx�A�f  "          @�=q��G�����@��B  C7#���G����
@��\B(�CB33                                    Bx�A�  
Z          @����
=�#�
@��BA�C5� ��
=��\)@�(�B7p�CE@                                     Bx�AͲ  	�          @�\)�����\)@��B�CCn����.�R@��B��CL��                                    Bx�A�X  .          @�33�Ϯ�˅@~�RA�(�CA�=�Ϯ�!�@]p�A��CIT{                                    Bx�A��  z          @������ÿ��@�G�A��C@  �����ff@dz�A��
CGǮ                                    Bx�A��  
�          @�����Q쿢�\@�p�BS=qCC�R��Q��0��@��B<  CSs3                                    Bx�BJ  "          @�z��p  ����@ƸRBe�CGs3�p  �4z�@�{BJ�CX�                                    Bx�B�  
�          @�\)��  ��@�p�BR�CO�H��  �`��@�
=B1�
C]E                                    Bx�B%�  T          @��
�u���@���BO(�CT���u�r�\@�Q�B+ffC`��                                    Bx�B4<  T          @����.{�'�@�  Bop�C_��.{��(�@��BCz�Cl�f                                    Bx�BB�  �          @�
=�9���\)@�\)Bq�HCY���9���p��@��BIz�ChL�                                    Bx�BQ�            @����X���Dz�@�z�BP��C^5��X�����@��RB&�HCh}q                                    Bx�B`.  
f          @�R����\)@�33B��\C]c����Z�H@�ffB_�Cn#�                                    Bx�Bn�  T          @�Q��`����H@�G�BQ�CV���`���l(�@�G�B-  Cbp�                                    Bx�B}z  
�          @߮����Tz�@��HB#��CZ��������\@i��A�\)Cb��                                    Bx�B�   "          @��
�C33�>{@�33B\p�C`8R�C33��33@��RB1Cj��                                    Bx�B��  �          @����~�R�k�@�\)B:=qC^��~�R��ff@�Bp�Cg0�                                    Bx�B�l  �          @�  �}p��E@��B:�\CY��}p�����@�p�B�HCc&f                                    Bx�B�  
Z          @����`�����@�BM=qCT{�`���W�@�  B+  C_�
                                    Bx�BƸ  T          @Ϯ�c33���@�BGCS��c33�P  @���B&�\C^u�                                    Bx�B�^  "          @�=q��p���H@�G�B'�HCP����p��[�@u�B	��CY�=                                    Bx�B�  �          @�G���ff�˅@j�HB��CH����ff�=q@L(�B=qCQ�
                                    Bx�B�  �          @������1녿����=qCJ�H���(���G��]�CH�                                    Bx�CP  T          @����P  ����@�G�Bv
=C;  �P  ��@���Bd��CQ{                                    Bx�C�  �          @�Q쿦ff?aG�@��B��\C�R��ff�k�@�B�33CWO\                                    Bx�C�  �          A=q���?�\@���B�p�C(��������@�
=B�
=C7�=                                    Bx�C-B  �          A
�H���H?�Q�AG�B�  C�=���H�J=qA{B�p�CU
                                    Bx�C;�  �          A33>L��?uA	��B��B�G�>L�Ϳ��A	p�B���C�b�                                    Bx�CJ�  
�          A
�H�@  ?c�
A	p�B��C!H�@  ��\)A��B�\Cl+�                                    Bx�CY4  
�          AQ�xQ�?fffA=qB�C	E�xQ쿇�A�B�p�Ccz�                                    Bx�Cg�  �          A�^�R>�AQ�B�8RCT{�^�R��=qA
=qB�p�Cq&f                                    Bx�Cv�  
          A
=�����AG�B�
=C9�)����
=qA	�B��CnǮ                                    Bx�C�&  �          AG��#�
=�A��B�ǮC)T{�#�
����AG�B�C{�                                    Bx�C��  
�          A���?�Q�A	p�B�B�C	#����\)A��B�G�C6�                                    Bx�C�r  T          A��ff?ǮAp�B��C�q��ff�   A�B�z�CE��                                    Bx�C�  
�          A
=��=q?aG�A\)B�z�CO\��=q��
=A�RB���CT�\                                    Bx�C��  �          Ap��^�R?n{A�B��)C�ÿ^�R����A\)B���Chn                                    Bx�C�d  �          A
�R��ff?�  A�B��HB��
��ff�W
=A
=qB�p�CL��                                    Bx�C�
  T          A��\)@
�HA��B�\B�zᾏ\)>�A�B��C0�                                    Bx�C�  �          A���HQ�@�H@ϮBj�CG��HQ�?fff@ۅB�B�C#�                                    Bx�C�V  T          @����G�@Z�H?�A>�\Cs3��G�@@��@	��A��C��                                    Bx�D�  �          @����  ?��R@�33B+��C#aH��  >�p�@��B5{C/�q                                    Bx�D�  T          @�\��\)>�=q?\)@���C1����\)>\)?(�@��C2�{                                    Bx�D&H  �          @�=q��\)>#�
��ff�s�
C2����\)>�������Z=qC1�\                                    Bx�D4�  "          @�G��ȣ׾�?�Q�AN�RC8aH�ȣ׿J=q?�ffA:{C;(�                                    Bx�DC�  "          @��H�o\)?\@�G�B`��C���o\)>��@�
=Bk��C1��                                    Bx�DR:  
�          @��
���\?^�R@��
BM(�C)�
���\�   @��BO
=C9�                                    Bx�D`�  �          @�p�����@�@�\)BS�C������?�G�@ӅBg�RC%�                                    Bx�Do�  "          @��
�a�@Q�@��BM��C	��a�?��@�(�Bk�\C�                                    Bx�D~,  
Z          A���Q���H@���B8\)CL���Q��j�H@�33B(�CV��                                    Bx�D��  
�          @�
=�A녿��
@�z�Bt�CR��A��HQ�@�33BUG�Ca�3                                    Bx�D�x  �          A(����S�
@���B.\)CSQ�����G�@�{B{C[�H                                    Bx�D�  
�          A������w
=@��B�CV�\�����Q�@g�Aң�C\��                                    Bx�D��  
f          A���޸R�$z�@�33A�z�CHB��޸R�`  @|(�A֣�CN�3                                    Bx�D�j  z          A
{��ff�
=q@���BffCH  ��ff�P  @�{B
�
CP�H                                    Bx�D�  
�          A
=����C�
@�\)BN�CW\�����@�p�B.�Cau�                                    Bx�D�  T          A�R���\�%�@�Q�BY�CTG����\�}p�@��B;33C`&f                                    Bx�D�\  �          @�33�����(�@N�RA�ffCI�����'
=@/\)A�\)CO��                                    Bx�E  �          @��R��{�
=q��\�N�RC7�q��{������W�C5)                                    Bx�E�  "          @�����z῾�R@"�\A�CG����z���R@
�HA��CM��                                    Bx�EN  
�          @���1��p�׿�G���
=Ciu��1��Vff�G���CfO\                                    Bx�E-�  .          A!p��@  ���AQ�B\��Cn���@  ��G�@ᙚB1�CuT{                                    Bx�E<�  �          A0���Q����A\)BX��Cok��Q����@�33B-��Cu�\                                    Bx�EK@  �          A0���U��  Az�BSp�Coٚ�U��ff@�(�B(p�Cu޸                                    Bx�EY�  |          A333����
=AQ�Bi  CxB�����\A�RB;�\C}n                                    Bx�Eh�  T          A:ff�(����p�A��Be�Cu  �(����\AffB9=qCz�                                    Bx�Ew2  
Z          A:�H�7
=���\A�HB`z�Csٚ�7
=���RA(�B4�\Cy��                                    Bx�E��  
�          A:�R�!��ۅA  BK��Cy�!��	�@���Bp�C}��                                    Bx�E�~  �          A)�1G����A
ffB[��Cr�f�1G���=q@�B0Q�Cx��                                    Bx�E�$  �          A)p��;�����ABQ�Cs�;���@�  B%��Cx@                                     Bx�E��  �          A%�:=q��(�A (�BJ��Cs�f�:=q����@�z�Bp�Cx}q                                    Bx�E�p  T          A'\)�,(�����@�(�BDQ�Cv���,(���  @�Bp�Cz�)                                    Bx�E�  T          A/
=�9�����
@��HB6(�Cw޸�9�����@�\)B
33C{O\                                    Bx�Eݼ  T          A ���O\)��(�@���B2(�Cs
�O\)��p�@�33B�RCw�                                    Bx�E�b  �          A33�c�
��(�@��BA��Cl�\�c�
�׮@���Bz�Cr0�                                    Bx�E�  T          A�H�r�\����@�(�BO=qCeJ=�r�\����@��HB*  Cl�
                                    Bx�F	�  
�          A���G��u�@�z�B;ffC]����G���\)@��BG�Ce=q                                    Bx�FT  �          A
=�i���^{@�p�Bb�HC_���i����{@ڏ\B@(�Ci�                                    Bx�F&�  
�          A  �w��6ff@�33BWCXW
�w���33@���B9�Cb��                                    Bx�F5�  �          @��\�����Q�@��B,=qCJ
=�����I��@���B�RCR�{                                    Bx�FDF  �          @�����R�(Q�@��RB,\)CN����R�j=q@��\BCW�                                    Bx�FR�  
�          A{�\�Tz�@��A��CP���\���@XQ�A�ffCU                                    Bx�Fa�  
�          A�\�����^�R@��B�CR���������
@�(�A��CX��                                    Bx�Fp8  �          AQ���  �o\)@�G�B;ffC]T{��  ��33@�{BG�Cd                                    Bx�F~�  T          A
ff�8Q��o\)@�
=BcG�Chz��8Q����\@�33B=�\Cp��                                    Bx�F��  
�          AQ��W��p��@��BY�Cd#��W����@�{B6
=Cl\)                                    Bx�F�*  T          Ap���ff�c�
@�ffB<��C\B���ff��z�@���B�Cc�H                                    Bx�F��  �          Aff�p���w
=@\BACa�\�p����
=@��RB��Ch�f                                    Bx�F�v  "          A���X�����@�G�B'��Cl�3�X����z�@�{Bz�Cq�                                    Bx�F�  T          A��hQ�����@��RB:Ce���hQ���33@���B\)Ck�{                                    Bx�F��  �          @�ff��Q����@�33B�HCb����Q����H@|��A��RCg�                                     Bx�F�h  T          @�z��������@�(�A�p�C^�{�����Q�@K�A�
=Cb�3                                    Bx�F�  �          @�p��|����\)@�BQ�Cc  �|����ff@S33A�  Cgp�                                    Bx�G�  
�          @�
=�i�����R@�=qB��Ci���i����z�@C33A�p�Cm)                                    Bx�GZ  T          @�\)�aG�����@c�
A�33Cl�)�aG���
=@{A�
=Cou�                                    Bx�G    �          @�ff�J�H���\@��B
=Cqk��J�H����@HQ�A��RCt�                                    Bx�G.�  "          A{�QG�����@��\B
=Co���QG���ff@k�A�Cs�                                    Bx�G=L  �          A   �5���\)@���B9p�Co{�5���\)@�G�Bp�Cs��                                    Bx�GK�  �          @����33����@�{BGCu�=�33���\@�\)B�
Cy�)                                    Bx�GZ�  �          @�  �&ff��@��HBB�RCn&f�&ff����@�
=B33Cs=q                                    Bx�Gi>  �          @�Q��1G���z�@��B5
=Cm���1G�����@��\B�CrL�                                    Bx�Gw�  
f          @����z���(�@��\B-�
Cu�)�z�����@�G�Bz�Cx��                                    Bx�G��  
          @����.{���
@�B5ffCo���.{����@��B�Cs�H                                    Bx�G�0  
(          @��
�1G���33@�z�B4Q�Cn���1G�����@�ffB�CsT{                                    Bx�G��  
�          @�Q�����p�@��HB9ffCvu�����=q@�z�B(�Cy�                                    Bx�G�|  T          @�p�?Q���
=@�RA��RC�Y�?Q���Q�?h��@�z�C�8R                                    Bx�G�"  �          @��R��\)��\@c�
A�G�C�޸��\)��@p�A�p�C���                                    Bx�G��  �          A�R��\)��{@�\)B(�C�ٚ��\)��=q@�(�A޸RC���                                    Bx�G�n  "          Aff>�ff��
=@���BDG�C�#�>�ff���
@��RB�C�˅                                    Bx�G�  �          A�
=L�����H@���Be  C�K�=L�����
@���B;p�C�:�                                    Bx�G��  �          Az�?Q��\��@�
=B��C���?Q�����@�
=B\C��
                                    Bx�H
`  �          A Q�?O\)�n{@���Bt�C�"�?O\)��z�@��
BL  C���                                    Bx�H  �          @��?�p��(�@�ffB_\)C��f?�p��I��@l��B;{C���                                    Bx�H'�  T          @��
@��
���R@w
=B%�C�@��
�
=q@a�B�C�33                                    Bx�H6R  "          @��\@@  �333@\)B0��C��H@@  �]p�@[�BQ�C�w
                                    Bx�HD�  �          @��?J=q��  @U�BC��)?J=q����@$z�A�p�C���                                    Bx�HS�  T          @�Q�?E�����@U�B�RC��R?E���p�@   A�G�C�t{                                    Bx�HbD  T          @љ�>�
=��G�@�ffB7��C��)>�
=��G�@u�BQ�C�=q                                    Bx�Hp�  �          @��;�Q���@��B>(�C��{��Q����R@�Q�B�C��                                    Bx�H�  T          @߮�����G�@�  BC�C��������z�@�(�B�RC��                                    Bx�H�6  �          @���>B�\���H@�  BA�C�>B�\��Q�@���B�HC���                                    Bx�H��  T          @��>B�\�j�H@z�HB;�\C���>B�\��G�@N�RB�C�J=                                    Bx�H��  "          @���#�
��Q�@���B3(�C�uÿ#�
��z�@Q�B��C�3                                    Bx�H�(  �          @�R��  ��{@�  B@�C�uþ�  ����@��B{C���                                    Bx�H��  �          A��33��Q�@\BD=qC�R��33��\)@��HB��C�c�                                    Bx�H�t  �          A=q��p����@��BI{Cw�)��p����
@�z�B$�C{aH                                    Bx�H�  T          A���(�����@��B?�C{�q��(���p�@�BG�C~�f                                    Bx�H��  �          A���$z����\@�p�BEffCv(��$z���  @�=qB!{Cy��                                    Bx�If  �          A#
=�Y������@�Q�BF��Co  �Y����z�@�{B$Q�Cs�                                     Bx�I  �          A#\)���\��\)@�G�BG=qCa����\��\)@�(�B)�RCh�=                                    Bx�I �  �          A!���
=��@��BR(�Cbc���
=���R@�(�B4ffCi�=                                    Bx�I/X  �          A�R�����G�@��RBU
=Cb33������@�(�B7ffCi��                                    Bx�I=�  T          A���
=�l(�@�p�BP�CY�q��
=��p�@�ffB6=qCb.                                    Bx�IL�  �          A\)���H��@�p�BH��CG33���H�O\)@�\)B9
=CQ
=                                    Bx�I[J  T          A�������@�G�BMffC>�����@�  BBCI��                                    Bx�Ii�  T          A�����5A ��BgQ�CV5�����(�@�ffBN��C`��                                    Bx�Ix�  �          AQ����{@��
B]�Csp����(�@׮B;33CxE                                    Bx�I�<  �          A���   ���@��HB]\)Cr�{�   ��G�@�{B;  Cw��                                    Bx�I��  �          A�������\)@��RBY��Cv�H����θR@߮B6ffCz��                                    Bx�I��  T          A���(��W�AQ�BdC[0���(���p�@�BJ��Cd�                                    Bx�I�.  �          A��>{��p�@�p�B9�Cs\)�>{��
=@��
B{Cv�                                    Bx�I��  �          A=q�!��Å@�B,\)Cw}q�!�����@��B	�Cz33                                    Bx�I�z  
�          AQ��L(����H@�B�CsT{�L(���z�@��\A�  Cu�                                    Bx�I�   �          A���s33��
=@љ�B5��Ci�3�s33�ƸR@��
B�RCn��                                    Bx�I��  
�          A������k�A ��Bt33C?������p�@���BgCN�)                                    Bx�I�l  
�          A����=q�%@�\)B>z�CM�3��=q�e@�\)B+��CV
=                                    Bx�J  T          A���  �33@�G�Be\)CPaH��  �^{@�=qBPC[:�                                    Bx�J�  �          A{�q녿�\A��B{{CM\�q��A�@�
=Bg�CZ                                    Bx�J(^  �          A���`  ��
=A33B��=CJ=q�`  �-p�@��
Br��CY�\                                    Bx�J7  
�          A�
�.�R?�ff@�B��C�q�.�R����@�Q�B��qC6�                                    Bx�JE�  "          @���W�@&ff@��HBN  C^��W�?�(�@�{Ba�C�                                    Bx�JTP  T          A�R��\)@Vff@��B�CǮ��\)@%�@��\BC��                                    Bx�Jb�  �          @����  @g
=@-p�A��\C�q��  @H��@O\)A���CO\                                    Bx�Jq�  "          A Q���p�@r�\@�\)B33Cp���p�@@  @�Q�B)�\C��                                    Bx�J�B  �          @�{���\@k�@�G�B{C�����\@8��@���B-=qC{                                    Bx�J��  �          @��R�$z�@����=q��
B噚�$z�@���Z=q��Q�B�B�                                    Bx�J��  �          @������
@��8Q쿳33Cp����
@���?
=q@�z�C�)                                    Bx�J�4  �          @�G���  @��׿�
=�r�HC5���  @�  ����(�C \                                    Bx�J��  �          @���1G�@�\)���H���B���1G�@�G��L���ͅB�G�                                    Bx�Jɀ  �          @���(�@����o\)��z�B�q�(�@���5���\)B�B�                                    Bx�J�&  �          @��\�8��@�����7\)B�u��8��@�{��{�G�B�=q                                    Bx�J��  �          @�  �p��Tz���z��R(�Cis3�p��p����H�lCa                                      Bx�J�r  �          @�ff��\�A������OffCl  ��\��\����k  Cd@                                     Bx�K  �          @�\)���\��=q�|�CZG����E�����HCI�f                                    Bx�K�  �          @�=q��  �u���
�P�C^���  �AG���z��r(�C{��                                    Bx�K!d  T          A�
�����
=���H� Cc0������s33��=q��CM�
                                    Bx�K0
  �          @��H���R�7
=�����uCk
���R��\)��Q���C_�                                    Bx�K>�  �          @����\)�����Q��L�C�/\�\)�z=q��p��nffC��)                                    Bx�KMV  �          @��?   ��z���G�� ��C��=?   �����=q�C  C��                                    Bx�K[�  �          @�ff�J=q��녿���x��C�)�J=q��=q������p�C��R                                    Bx�Kj�  �          A�׿����@e�A�Q�C�� ������@�RAV{C��                                    Bx�KyH  �          A�׾�z����@(�Aj=qC����z��?��@�C�                                      Bx�K��  �          A+���� ��@Z�HA��C��
���'
=?�z�A$z�C�q                                    Bx�K��  
5          A3\)�У��&�\@qG�A��C����У��-��@p�A5�C���                                    Bx�K�:  G          A'\)�8����
@�Q�A�Q�C|&f�8�����@I��A�Q�C}#�                                    Bx�K��  
�          A(Q��z�H�(�@�
=A�Ct���z�H��H@}p�A��HCvJ=                                    Bx�K  
�          A������G�@��B(�RC^�H������\@��B�Ccs3                                    Bx�K�,  �          @����
=@ҏ\?c�
@��B�aH�
=@�z�?�=qAqG�B܀                                     Bx�K��  �          A"�R����A��<�����HB�\)����A33���
�
�\B���                                    Bx�K�x  �          A�\��Q�@��;����RB�𤿘Q�@�R����_
=B�{                                    Bx�K�  �          @�  @z�@�{�Q����B���@z�@�
=�����d��B��)                                    Bx�L�  �          @�?˅@��׿��H�>=qB���?˅@�z�����Q�B�p�                                    Bx�Lj  "          @�����@.{?�
=A�  Cn���@�H@�\B�HC�=                                    Bx�L)  �          @����z�@�>�@_\)Bң׿z�@�>�(�A5p�B�33                                    Bx�L7�  �          @��ÿxQ�@H���Q��3�
B�G��xQ�@dz��3�
���B�Q�                                    Bx�LF\  �          @ҏ\�8Q�@�z�=#�
>�33B�G��8Q�@\?fffA
=B�Q�                                    Bx�LU  �          AG��p��Q�@���B��3C\�{�p��C�
@��Bj
=Cg(�                                    Bx�Lc�  �          Ap����*=q@�  B�ffCec����i��@ٙ�Bf��Cm��                                    Bx�LrN  �          @�녿���?�{@z�B;p�CE����?O\)@p�BK�C0�                                    Bx�L��  �          A�
�(��@�����z���z�B��H�(��@����I����z�B�8R                                    Bx�L��  �          @�z��\)@�\)���\���Bި��\)@���o\)��B�
=                                    Bx�L�@  �          A33�'�@�(���z��;ffB���'�@���(���B��                                    Bx�L��  T          Ap����@+����Q�B�8R���@l(����
�v��B�
=                                    Bx�L��  T          Azᾔz�@l����Hp�B��)��z�@��\���H�iffB���                                    Bx�L�2  �          @�Q쿌��?��P���tz�B������?����A��ZffB�                                    Bx�L��  �          @�p��L��?ٙ�������B� �L��@G���Q��l=qB��                                    Bx�L�~  	�          A	���ff?0���{k�C𤿦ff?�����R�)B���                                    Bx�L�$  
Z          A8�ͿB�\@5��1�\)B�녿B�\@���)G��B��                                    Bx�M�  
�          A:=q��33@�{�!��}�\B�8R��33@�Q���_�B���                                    Bx�Mp  
Z          @�녿��
?�  ��
=�C�����
?Ǯ�����B��                                    Bx�M"  
�          @�\)�5��Q�@��\B'\)ClW
�5���H@~{B(�Co�)                                    Bx�M0�  	�          A"�R���\��p�@�p�B
=Cmu����\���H@��A�p�Co��                                    Bx�M?b  	�          A{�p  ��ff@�Q�B*
=Cl�3�p  �Ϯ@�{B�HCp                                      Bx�MN  
          @�  >��
@�Q��0���	�
B�  >��
@�33����֣�B��                                    Bx�M\�  
(          @�Q�>�33@=q���n��B�p�>�33@<���tz��P�B�k�                                    Bx�MkT  
�          A  �}p�@�33���C(�B��}p�@��H��ff�%33B�33                                    Bx�My�  
�          A�ÿ
=q@��H��
=�:��B����
=q@��
�����ffB�ff                                    Bx�M��  
�          A\)��  @����p��=  B�LͿ�  @�G�������B��)                                    Bx�M�F  
�          A!���A�R�XQ���{B�Ǯ��A���\)�O�B��
                                    Bx�M��  
�          A"�H�Dz�AQ쿚�H��p�B׽q�Dz�A��=#�
>uB�z�                                    Bx�M��  
Z          A'
=��  A���G��2�\B�\��  A�
�J=q��=qB�8R                                    Bx�M�8  �          A/���  Aff�   �S�
B�
=��  A=q��G����
B���                                    Bx�M��  �          A8����{A&�\�
=�<��B�8R��{A*{�}p���p�B�aH                                    Bx�M��  T          A$���S33A(����
=B�=q�S33Aff��(��=qBٽq                                    Bx�M�*  
Z          A(�>��HA=q>8Q�?��
B�#�>��HA ��?�p�A
�HB�\                                    Bx�M��  
Z          A(�@��@�G�@*=qA�G�Ba(�@��@�z�@^{A�  BZz�                                    Bx�Nv  
�          A�=#�
@��?�ffAW�B�=q=#�
@�{?�p�A���B�8R                                    Bx�N  �          A6�\��p�A�����(�B�����p�A�
�5�dz�B���                                    Bx�N)�  �          @���[�@�ff>���@G�B�=q�[�@�33?�33A��B��H                                    Bx�N8h  
�          @�G�?:�H���R�8Q��p33C�q?:�H��{�C33��C���                                    Bx�NG  
�          @�������z=q>�{@�z�C�������z�H�#�
��C��\                                    Bx�NU�  
          A=p�?˅�7
=?��@.�RC��3?˅�6�R�fff��
=C��{                                    Bx�NdZ  	�          AE�?��R�B{�aG����C���?��R�>�H����5�C���                                    Bx�Ns   
�          ADQ�?��H�4��?�G�A�C���?��H�7
=>��
?���C���                                    Bx�N��  
P          A1p�@����@N�RA��C�4{@����ff@  AP��C���                                    Bx�N�L  
!          A�H@�����R@Z=qA�\)C�j=@����@�HAmp�C��)                                    Bx�N��  
�          A ��?���@)��A�G�C��3?���?�G�A{C���                                    Bx�N��  
�          A0�Ϳ�ff�  @  AMp�C�|)��ff�33?��\@���C��3                                    Bx�N�>  
�          A-?�=q�#33@J�HA��\C�h�?�=q�((�?��A z�C�G�                                    Bx�N��  �          ALQ�@\(��<��@]p�A{\)C�!H@\(��B{?�(�A{C��                                    Bx�Nي  T          AQp�@����:=q@N�RAd��C���@����?33?�\@���C�c�                                    Bx�N�0  |          A\z�@�ff�DQ�@l(�Ay�C��q@�ff�J{@	��A  C��{                                    Bx�N��  �          Aa�@��\�I�@L(�AQC�t{@��\�N�R?�\)@ӅC�8R                                    Bx�O|  �          A\��@�{�B�R@r�\A�C���@�{�H��@G�A�
C�:�                                    Bx�O"  f          ALQ�@>�R�6�R@�p�A��RC�S3@>�R�>ff@O\)Al��C��                                    Bx�O"�  	`          AG33@tz��3�
@FffAjffC�c�@tz��8��?��HA Q�C�*=                                    Bx�O1n  @          A0Q�@s�
�!?�ff@�(�C�S3@s�
�#33=�\)>�{C�@                                     Bx�O@  �          A33@0����R@h��A�z�C�
@0�����@%A{
=C��R                                    Bx�ON�  �          A
=?@  ��@��
B�C���?@  ��\)@�{A�\C��\                                    Bx�O]`  "          AQ�=�\)��z�@�p�B4z�C�K�=�\)��p�@�G�B��C�C�                                    Bx�Ol  �          A�>�ff��ff@�
=B<��C��q>�ff��  @���B!{C��f                                    Bx�Oz�  �          A�@��Q�@C�
A��HC�+�@���?�A2{C��)                                    Bx�O�R  
�          A#33@E���@XQ�A���C�b�@E�
=@G�AMC��                                    Bx�O��  
�          A�H@J=q�Q�@+�AzffC��f@J=q�Q�?�=qA�RC�g�                                    Bx�O��  
�          A{@6ff�=q@,(�A��HC��@6ff�ff?��HA.ffC�\)                                    Bx�O�D  
�          A(��@=p���H@>�RA�G�C�� @=p���?�A{C�E                                    Bx�O��  
�          A*�R@s33���@3�
AtQ�C��R@s33��?�z�A
=C��{                                    Bx�OҐ  "          A�H@j=q��H@ ��AiG�C�&f@j=q��\?�Q�A��C���                                    Bx�O�6  �          Aff@\)��xQ����HC���@\)��H���EC���                                    Bx�O��  �          A ��@w����=#�
>aG�C���@w�����  ��ffC��{                                    Bx�O��  �          A'33@�33��H?У�AQ�C��q@�33���?�@J=qC���                                    Bx�P(  "          A%�@���
�R?Y��@�  C��)@���\)�.{�xQ�C���                                    Bx�P�  "          A&=q@��\�	?�G�@�Q�C�u�@��\�33>�  ?�\)C�T{                                    Bx�P*t  /          A(z�@�����\?z�H@��\C�` @����\)�#�
��C�G�                                    Bx�P9  
          A z�@Å��p�>Ǯ@\)C���@Å������B�\C�ٚ                                    Bx�PG�  �          A(�@�����{�(��x��C�ٚ@�����=q�����
=C��                                    Bx�PVf  �          A
�\@�
=�����Ǯ�&=qC��@�
=���\�{�m�C��                                    Bx�Pe  �          A(�@�{��ff���H�3�C�AH@�{���R�ff�y�C�ٚ                                    Bx�Ps�  
�          A(�@�ff������
=�M�C�>�@�ff��G��'����C��H                                    Bx�P�X  "          AQ�@`����\)�AG���=qC�8R@`�����H�o\)��Q�C���                                    Bx�P��  T          A�@%���(��=q���HC��\@%�����Mp���Q�C�^�                                    Bx�P��  �          A
�\@XQ���{�:�H���HC���@XQ���녿У��.{C���                                    Bx�P�J  �          Ap�@H�������
�   C�t{@H����R���\��(�C���                                    Bx�P��  "          A=q@J=q���������C���@J=q�陚��z�� ��C��{                                    Bx�P˖  �          AG�@G���G�?���A�
C�Ff@G����
>��R@
�HC�33                                    Bx�P�<  �          A(�?���\)@0��A��C�q?���  ?���Ao�
C��                                    Bx�P��  T          A�׼#�
��z�?�(�AE�C����#�
� ��?Q�@�=qC���                                    Bx�P��  "          A��z����@.{A�Cu��z��  ?�ffA?
=C�f                                    Bx�Q.  �          A�
�5��(�@k�A�p�Cx�R�5��  @4z�A���Cy��                                    Bx�Q�  T          @�ff�z���(�@e�A޸RCz  �z��׮@5A�  Cz�q                                    Bx�Q#z  �          A	p�����ָR@�G�B  CzQ������@p��A�33C{��                                    Bx�Q2   �          A
�H�(����
=@   A`Q�C����(���=q?��@�33C��\                                    Bx�Q@�  �          A'33@�\)��
�\)�E�C��@�\)�녿Ǯ�	�C�Ff                                    Bx�QOl  �          A�H@��
��{�#�
�W
=C�
=@��
��z�k�����C�!H                                    Bx�Q^  h          AQ�?��
���H@���BD�C�k�?��
��ff@��RB+�C��q                                    Bx�Ql�  �          A(��ٙ��p  A�
B}�RCu�3�ٙ�����@���Bf
=Cy��                                    Bx�Q{^  "          A���<���QG�A��B}�Cd  �<����\)A{Bi��Ck#�                                    Bx�Q�  a          A (���ff��A��Bv  CPxR��ff�O\)A�
Bh{CY��                                    Bx�Q��  T          A&{�r�\�Mp�A��Bu�\C\@ �r�\��{A
=Bc�
Cc��                                    Bx�Q�P  o          A(���#33��33A�Bup�Co���#33���HA
=B_  Ctz�                                    Bx�Q��  
�          A{�c�
���@�
=Bo
=C�y��c�
���@�p�BV�C�S3                                    Bx�QĜ  "          Aff���R����A�Bx
=C��f���R��z�@�B^��C�AH                                    Bx�Q�B  �          A(������hQ�A
=B��{C}.������G�@��Bn��C�33                                    Bx�Q��  
�          A%���(��!�A�B�z�Ch+���(��eA=qB��Cq=q                                    Bx�Q��  �          A$z��)����p�A  B���CL���)����\A��B��3C\�{                                    Bx�Q�4  �          A'\)�<(�?O\)A   B�{C$���<(���z�A Q�B�(�C9�                                    Bx�R�  �          A(  �,��?�ffA!G�B��HC���,�ͽL��A"{B��C5#�                                    Bx�R�  �          A"ff�R�\?s33A�B�=qC#�R�R�\��Q�A(�B��
C5�)                                    Bx�R+&  �          @��?��R���@�Q�B%z�C�%?��R��33@p��B�C�*=                                    Bx�R9�  �          A	G�@>{����@\��A�33C���@>{��@+�A��C�#�                                    Bx�RHr  "          Ap�@���߮@�z�B�C�~�@����\)@�33A܏\C���                                    Bx�RW  
�          A\)@H����z�@�G�A�z�C�=q@H�����@aG�A���C���                                    Bx�Re�  G          Aff@|(��У�@`  A�p�C��3@|(��ۅ@1G�A���C��                                    Bx�Rtd  T          A�@��\��?�z�@��
C�ff@��\��Q�>��
@�C�E                                    Bx�R�
  f          A�@�����?�{@�=qC��R@����(�>��?��C��
                                    Bx�R��  �          A"=q@�{��R?!G�@e�C�p�@�{�
=��\)��{C�h�                                    Bx�R�V  �          A=q@��
����.{�xQ�C�\)@��
� �׿����G�C�xR                                    Bx�R��  �          A�\@ȣ������\)�
=C��
@ȣ���\)�Q��L��C��R                                    Bx�R��  �          A�@�\)��  ����� ��C�g�@�\)�����H��C���                                    Bx�R�H  �          A!�@Ӆ��ff���  C��=@Ӆ��\)�p��a�C�8R                                    Bx�R��  �          A'
=@���(����H�,(�C�(�@����
�333�v�RC��H                                    Bx�R�  a          A)�@�p�������%��C�  @�p����
�)���k\)C���                                    Bx�R�:  =          A(z�A
=q��(���=q���C�� A
=q��\)�ٙ��G�C�8R                                    Bx�S�  T          A�
A���ff���+\)C�%A���\)�G��X  C���                                    Bx�S�  �          A��@��R�333��ff�ݮC�z�@��R��  ��  ���\C��                                    Bx�S$,  �          A
�\@����y�����HC��@�����|(����C�s3                                    Bx�S2�  �          A�
@�=q>��qG����
?u@�=q?��o\)��@~�R                                    Bx�SAx  �          A�R@��
>Ǯ�Tz���{@;�@��
?=p��P����z�@���                                    Bx�SP  �          @�
=@���@���;���  A�{@���@+��*=q����A�                                      Bx�S^�  T          @��@��@6ff�=q����A�  @��@E����RA��
                                    Bx�Smj  �          A Q�@�@(���p���A�ff@�@7
=���\��A��\                                    Bx�S|  T          AQ�@�@AG���\�d��A���@�@N{���H�?33A�33                                    Bx�S��  T          A=q@���@\���
=�u�AԸR@���@j=q�޸R�H��A�p�                                    Bx�S�\  �          AQ�@أ�@mp������A��
@أ�@{�����W�A��                                    Bx�S�  �          A��@�Q�@��
�%����Bff@�Q�@���
=�l��BQ�                                    Bx�S��  �          A�@�
=@��
�#33��G�A�(�@�
=@��
�ff�j�HB{                                    Bx�S�N  T          A�@�\)@����I����\)B��@�\)@��R�*�H����BQ�                                    Bx�S��  �          Az�@���@�{������Q�B2=q@���@���o\)��B;��                                    Bx�S�  �          Az�@��@�G������$��B#��@��@��\��ff��\B2�                                    Bx�S�@  �          A�R@�Q�@�  ��\��z�B33@�Q�@�
=��  �FffB#�                                    Bx�S��  �          A33@��R@��\�(����=qB'�@��R@��\�z��c\)B,Q�                                    Bx�T�  �          A=q@���@����Z=q��Q�B\)@���@�33�<(���B��                                    Bx�T2  �          A�H@���@[��0  ��33A���@���@l���Q����RA�33                                    Bx�T+�  �          A ��@�33@��H�H����\)B(�@�33@�z��,�����B��                                    Bx�T:~  �          A�@�
=@)���g�����A�ff@�
=@AG��Tz���{A��                                    Bx�TI$  �          A�\@�ff@\(��XQ���Q�A��\@�ff@q��@  ���HA�R                                    Bx�TW�  T          A
=@�p�@z�H�p  ��  A��@�p�@�G��Tz���{B�                                    Bx�Tfp  �          A
=@���@\)��  ��G�A�(�@���@:�H�}p���  A���                                    Bx�Tu  �          A
{@߮?�=q��
=��Ak
=@߮@z���  ��\)A���                                    Bx�T��  �          A��@�?z�����	��@��R@�?�{�����{A                                      Bx�T�b  �          A�@�׽���p���z�C��\@��>�p��������@5                                    Bx�T�  �          A33@�Q�����=q���HC�J=@�Q��������  C��                                    Bx�T��  �          Aff@�=q�E���Q���C��R@�=q�"�\�����=qC�Ǯ                                    Bx�T�T  �          Az�@�p��/\)�����=qC��)@�p��
�H�������C���                                    Bx�T��  �          A��@������������C�P�@���c33����	��C�
=                                    Bx�T۠  �          A�\@�  �QG��������C��q@�  �1G���(�� ��C���                                    Bx�T�F  �          A��@�R��R��\)���C�c�@�R�ٙ���{��C�Z�                                    Bx�T��  �          A�@��ÿ����\)�G�C��@��ÿ��\���
�p�C��                                    Bx�U�  �          Aff@���z������z�C�=q@��W
=�����Q�C���                                    Bx�U8  a          A
ff@�  ������{�{C�XR@�  �L�������
C��q                                    Bx�U$�  y          Aff@���*=q��=q�z�C��{@�������=q�  C��H                                    Bx�U3�  �          AQ�@�Q콣�
�����=qC��@�Q�>�
=��  ��@R�\                                    Bx�UB*  �          Aff@��H@w
=��ff�ܣ�A�@��H@���������A�z�                                    Bx�UP�  �          A�A
=@U��������A�\)A
=@o\)�k����A�z�                                    Bx�U_v  |          A!�A@:�H�z=q��  A�A@S�
�e��(�A��
                                    Bx�Un  �          A (�A	�@U��y�����A�{A	�@n{�a����A��
                                    Bx�U|�  �          A{A(�@xQ��hQ����A��A(�@�\)�Mp����\A�
=                                    Bx�U�h  �          Ap�@�ff@�\)���\�ѮA�=q@�ff@���w
=���A�\)                                    Bx�U�  �          A%p�AQ�@�  ����A�=qAQ�@���l(���{A�p�                                    Bx�U��  �          A'�
@��R@��R�������B@��R@�z��u��  B=q                                    Bx�U�Z  �          A%@���@�ff������B  @���@����������\B��                                    Bx�U�   �          A��@��?�Q��w
=�أ�A0(�@��?��l(���Q�A^�R                                    Bx�UԦ  �          A	G�@��@�H�p����(�A��\@��@333�^�R��  A��                                    Bx�U�L  �          Aff@�\@U��P����  A���@�\@i���9����  Aͮ                                    Bx�U��  �          A�@��R@Q��QG���(�A�33@��R@g
=�:=q���A��H                                    Bx�V �  �          A��@��@'��[���p�A��@��@=p��H������A��R                                    Bx�V>  �          A�A�@�R��G����Aw\)A�@(���q���{A�
=                                    Bx�V�  �          Ap�@��?�(��s�
��{ABff@��@
=�g
=���HAl��                                    Bx�V,�  �          A
=A�H?�ff�Y�����@�=qA�H?�33�QG���  A\)                                    Bx�V;0  �          A
{@�Q�?�p��e��Q�A  @�Q�?����\(����A:{                                    Bx�VI�  �          A\)@�\?G��c�
��{@��@�\?�33�]p����
A=q                                    Bx�VX|  �          @�{@�33�k��h����(�C��R@�33>���h����ff?��                                    Bx�Vg"  �          A�\��\)��?�  @�Q�C��q��\)���>aG�?�ffC�                                    Bx�Vu�  �          A$  �5�#\)>�Q�@ ��C�  �5�#
=�5��  C���                                    Bx�V�n  �          A((��+��'33>�G�@�HC�*=�+��'
=�(���fffC�*=                                    Bx�V�  T          A+��z��+\)=#�
>k�C�p��z��*ff��=q��G�C�n                                    Bx�V��  �          A%�����$�þǮ��C��)���#\)��p��\)C���                                    Bx�V�`  �          A"�\>�=q� �ͿE����
C�Ǯ>�=q��R�����&�\C��=                                    Bx�V�  �          A��@Z�H�	G�����8  C�� @Z�H�p��1G�����C�*=                                    Bx�Vͬ  �          A��@��\��  �#33�p��C��\@��\��{�U���Q�C��f                                    Bx�V�R  �          A{@����  �%�s
=C��3@�����X�����\C�c�                                    Bx�V��  �          A33?���Q�+�����C�n?����\����&�RC��                                    Bx�V��  �          A%����{��=q@~�RA���C_p���{�޸R@P��A��RCa\                                    Bx�WD  �          A0������
=@�G�A���CX������p�@x��A�G�CZ��                                    Bx�W�  �          A1G�� Q����@�z�A��CXaH� Q���z�@��A��CZ�\                                    Bx�W%�  �          A0Q���
���@���A��
CS����
��ff@�=qA��CU�H                                    Bx�W46  �          A.�R�����33@��A�CPT{�����33@�
=A��CR�H                                    Bx�WB�  �          A,��������@���A�CL�����G�@�=qA�ffCOL�                                    Bx�WQ�  T          A-p��{�e�@��A�G�CIk��{����@�ffA�ffCK�                                    Bx�W`(  �          A,����R�U@�=qAȣ�CH��R�s�
@�{A��HCJ��                                    Bx�Wn�  �          A,Q��
=�8Q�@��A��HCD���
=�Tz�@~�RA���CGc�                                    Bx�W}t  �          A,�����AG�@�G�A��CE����^�R@�ffA���CH�                                     Bx�W�  �          A-G�����@  @���A�G�CEٚ����^�R@���A��CHz�                                    Bx�W��  �          A-G��
=�(Q�@��
A�  CC���
=�Fff@��A�CF5�                                    Bx�W�f  �          A(z��ff���@��
A�Q�C?���ff���@�z�A�33CB��                                    Bx�W�  �          A$(���\��p�@�ffA��C;ٚ��\��  @�G�Aޏ\C?�                                    Bx�WƲ  �          A,���=q�8Q�@�\)A�{CE}q�=q�W�@�z�A�(�CH=q                                    Bx�W�X  �          A0���z��N{@�Q�A�\)CF���z��k�@�z�A��\CI#�                                    Bx�W��  �          A1���Ϳ�p�@�=qA�C?k�����\)@��\A¸RCB=q                                    Bx�W�  T          A3\)��Ϳ�p�@���Aڣ�C>������@�AиRCA
                                    Bx�XJ  T          A2=q��R��@�G�A��C7\��R����@�ffA��C:(�                                    Bx�X�  �          A.�R�����@_\)A��
Cp��������@   AT��Cqp�                                    Bx�X�  �          A.�R���R�
=@@  A\)Cn�)���R�\)@ ��A*=qCo^�                                    Bx�X-<  T          A/33��z��{@eA�33Cg�\��z���@,(�Ac�Ch�3                                    Bx�X;�  �          A/33�ڏ\��@��RA��Cck��ڏ\���H@XQ�A�  Cd��                                    Bx�XJ�  �          A.�R��ff��=q@�p�A��C`����ff��@W�A�  Cb!H                                    Bx�XY.  T          A/�
�{���
@��
A�z�CP�)�{��@��HA�G�CS�3                                    Bx�Xg�  �          A/�
���
���H@��A��HCY�q���
���@z�HA��C[��                                    Bx�Xvz  T          A/����H��  @�
=A��CZ�\���H��ff@p��A�  C\�                                    Bx�X�   �          A0����=q����@�
=A�
=C[T{��=q�ۅ@p  A��\C]B�                                    Bx�X��  �          A2{��p���ff@�
=A�C]+���p�����@mp�A�  C_                                    Bx�X�l  T          A1���33��G�@�A�{C]�=��33��@j=qA��C_��                                    Bx�X�  
�          A0����R��@_\)A��
CbQ���R��z�@(��A]�Cc��                                    Bx�X��  �          A-������Q�?:�H@y��Cl�=�����;����33Cl޸                                    Bx�X�^  T          A,����\)��H?���AffCf�=��\)�G�?n{@��Cg�                                    Bx�X�  �          A,����=q���
@3�
Aqp�Ce
��=q�{?�A$(�Cf                                    Bx�X�  �          A,Q��޸R��Q�@r�\A��Cb5��޸R��z�@=p�A�
Cc�f                                    Bx�X�P  �          A.=q�陚���H@u�A�33C`33�陚��\)@AG�A�33Ca�3                                    Bx�Y�  �          A0����\)���H@�33A���C\xR��\)����@eA�C^O\                                    Bx�Y�  �          A2=q�����p�@�33AΣ�CY�3�����p�@�z�A�ffC\#�                                    Bx�Y&B  �          A1�����ҏ\@���A���C\��������@h��A���C]�R                                    Bx�Y4�  �          A1��\)���@vffA��\C[�H��\)�ᙚ@E�A�
=C]z�                                    Bx�YC�  �          A0z��������H@qG�A�(�C]Y�������
=@>�RA{\)C^��                                    Bx�YR4  �          A/�
��Q���Q�@XQ�A��C^���Q����H@$z�AXQ�C_n                                    Bx�Y`�  �          A0(������ff@�p�A��CYn�����(�@\��A��
C[L�                                    Bx�Yo�  �          A.�\�(��|(�@�=qA�CLٚ�(���G�@��HAޣ�CP�                                    Bx�Y~&  T          A-G���
�y��@�\)A���CL����
��  @�  A��CO��                                    Bx�Y��  �          A-p��{�|(�@�(�A�z�CM33�{���@���A���CP��                                    Bx�Y�r  �          A+�
���x��@�ffB =qCMJ=����Q�@�
=A���CP�R                                    Bx�Y�  �          A,����R�dz�@��RA��CJ�q��R��ff@���A�  CN��                                    Bx�Y��  �          A,������k�@���B=qCK�=������@��A��
COT{                                    Bx�Y�d  �          A,  � �����@�{A�CO��� �����@��A�  CS{                                    Bx�Y�
  �          A*�\���R���\@���A�CQ����R��p�@��HA�=qCU                                      Bx�Y�  �          A)�����{@�z�A�G�CQ8R������@��HA�=qCT\)                                    Bx�Y�V  �          A"=q����Q�@s33A��HCX)�����@H��A�
=CZ
=                                    Bx�Z�  �          A!����\��z�@z�HA�
=CWh���\����@Q�A���CYp�                                    Bx�Z�  �          A33��p���{@XQ�A��CYu���p���G�@,��A|z�C[#�                                    Bx�ZH  �          A�\������p�@@��A��C\������
=@�AT��C^#�                                    Bx�Z-�  �          A��������33@2�\A�{C\xR������(�@�
AB{C]�                                     Bx�Z<�  �          A����p���\)@=p�A���C[ٚ��p���G�@  AT(�C]@                                     Bx�ZK:  �          A��������@Y��A��C[:�������
@,��A�
=C\�                                    Bx�ZY�  �          A����Ǯ@P��A��RC_
=���ҏ\@   Ao33C`��                                    Bx�Zh�  �          A�������@AG�A�z�CZ� ������  @A`��C\�                                    Bx�Zw,  �          AG���33����@VffA��C\=q��33����@)��A�=qC]��                                    Bx�Z��  �          A����
=���@H��A�\)C[u���
=��=q@(�Al��C]
=                                    Bx�Z�x  �          A��޸R��ff@<��A�\)C[W
�޸R��Q�@��A\��C\��                                    Bx�Z�  �          A����
��Q�@p�Aqp�C]33���
��Q�?�p�A(��C^Y�                                    Bx�Z��  �          A���
���@G
=A��CZ�f���
��=q@(�Ar�\C\J=                                    Bx�Z�j  �          A����  ��=q@#�
A�Q�C\���  �\?���A8Q�C^                                    Bx�Z�  �          A��������H@�RAw\)C\�H������H?�\A/
=C]�{                                    Bx�Zݶ  �          A��������@ffAk�
C\����������?��A#�C]�q                                    Bx�Z�\  
�          A����  ��  @AO�
C[^���  ��ff?��A	G�C\aH                                    Bx�Z�  �          A�H��=q��?��A$(�C\�R��=q�\?h��@�ffC]�3                                    Bx�[	�  �          A�H�ٙ����?��A3�C\���ٙ��\?��@���C]�=                                    Bx�[N  �          A
=�Ӆ�ə�?fff@�33C_�H�Ӆ���
>�?L��C_��                                    Bx�[&�  �          A���p���\)@(�A[\)C]�f��p���ff?���AQ�C^�                                    Bx�[5�  �          A��У����?У�A#33C`��У��θR?\(�@�=qC`��                                    Bx�[D@  �          AG���G�����@(��A��C\W
��G���G�?�A>ffC]�                                    Bx�[R�  �          A  �����
=?�z�A>�HC_xR������?��@��C`Q�                                    Bx�[a�  �          Az���\)��33?�z�A=��C^.��\)����?�33@��HC_
=                                    Bx�[p2  �          A33���
��p�?���A(�C_����
��G�?
=@j�HC_�\                                    Bx�[~�  �          A�
�ᙚ��
=?�\)A@��CX���ᙚ���?�(�@���CY�                                     Bx�[�~  �          A  ��(���?��
A6�HCX��(����
?���@�RCX��                                    Bx�[�$  T          AQ����
����@  Ah  CW.���
��Q�?�{A%p�CXs3                                    Bx�[��  �          A����{���H?��A��Ch���{��ff>�
=@+�Ch�                                    Bx�[�p  �          A  ���H��{?�ffA�HCb�)���H�ҏ\?=p�@�ffCc=q                                    Bx�[�  �          A��ۅ��p�@{A��\CY��ۅ��{?�A:�HCZk�                                    Bx�[ּ  �          A����{��p�?Y��@�=qC^����{��\)=�G�?0��C^�
                                    Bx�[�b  �          A�������?�Q�A�CX޸�����=q?G�@��CY��                                    Bx�[�  �          A	���p���\)?�z�A2{C\p���p���z�?u@���C]J=                                    Bx�\�  �          A�
���
��  ?�  @ٙ�C\Ǯ���
���\>���@�C]33                                    Bx�\T  �          A\)���
��Q�?fff@�(�C_:����
���\>.{?���C_��                                    Bx�\�  �          A=q�׮��Q�?xQ�@ڏ\CUǮ�׮��33>Ǯ@/\)CVJ=                                    Bx�\.�  �          A=q� �þu@�\A}�C5��� �ÿ�@\)Aw�C7Ǯ                                    Bx�\=F  �          Ap�� (��u@�\Ac
=C:�{� (����H?�33AS33C<�)                                    Bx�\K�  �          A��������\@G�Aa�CB�q������\?޸RAB=qCDff                                    Bx�\Z�  T          A�����
�'�?�
=AW33CG  ���
�6ff?���A/33CH�                                     Bx�\i8  �          A��
=�u?��A5p�CP���
=����?���@�G�CQ)                                    Bx�\w�  �          A����z��r�\?�G�AD��CO����z��\)?�G�A�
CQ0�                                    Bx�\��  �          AG������h��?���A(�CN+������p��?5@�p�CN�                                    Bx�\�*  �          A
=��Q��L��>���@5�CK���Q��N{���
��CK:�                                    Bx�\��  �          AG���������h���ϮCEh�������׿�(��
�HCDs3                                    Bx�\�v  �          A ��������Ϳ�\�K\)CDO\���ÿ�����\�k�CB��                                    Bx�\�  �          A=q����1G���{��(�CH0�����&ff���H�&�RCG
=                                    Bx�\��  �          A  ��(��Fff<�>W
=CJ)��(��E��\�(Q�CI��                                    Bx�\�h  �          A�
��ff�^{>��H@[�CM  ��ff�`��<�>L��CM8R                                    Bx�\�  �          AQ���ff�}p�?O\)@��
CP�\��ff����>�\)?�CQ@                                     Bx�\��  a          A(����
��{?��
A+
=CWQ����
��33?c�
@�
=CX=q                                    Bx�]
Z  
�          A������Q�?�(�A"�HCW�������p�?Q�@�p�CXn                                    Bx�]   �          A����
��33?�33A6ffCX=q���
����?}p�@ٙ�CY:�                                    Bx�]'�  �          A����(����H?��\A��CX
��(����R?(�@�ffCX��                                    Bx�]6L  T          A����G����?�\)A4��CQ�3��G����?��@�33CS�                                    Bx�]D�  �          A�
����~{?�\)A5G�CQxR�������?���@�{CR�{                                    Bx�]S�  �          A���\��z�?���A=qCRQ���\����?@  @���CS(�                                    Bx�]b>  �          AQ���33��33?��@�RCR���33���R>��@P��CR�f                                    Bx�]p�  �          AQ���=q��
=?333@�=qCRٚ��=q����=�G�?G�CS.                                    Bx�]�  �          A
=�ᙚ��  ?��@�ffCQ�=�ᙚ��33?   @aG�CR33                                    Bx�]�0  �          A��ff��  ?���Az�CQ�f��ff���
?#�
@���CR�                                    Bx�]��  �          A=q�޸R��z�?J=q@��CR�޸R���R>W
=?�p�CS.                                    Bx�]�|  �          A���\�n�R?�{A�COǮ��\�xQ�?Tz�@�33CP��                                    Bx�]�"  �          A ��������p�?�33A=CT��������?�ff@�G�CUB�                                    Bx�]��  �          A Q���=q��  ?ٙ�AC�
CT����=q��ff?��@��\CV{                                    Bx�]�n  �          @��H�Ϯ��  ?���AV�RCS���Ϯ���R?��RA{CT��                                    Bx�]�  �          @�=q��33��(�?���Af�RCU���33���?���A�HCV�                                     Bx�]��  �          @��������x��@�
Au��CSJ=������z�?��RA0��CT�f                                    Bx�^`  �          @��R��(��i��@z�A�(�CQ���(��|(�?��
AU�CS�3                                    Bx�^  �          @��R�\��33@��A�ffCV��\����?��
AV{CW��                                    Bx�^ �  �          @�\)��=q���H@&ffA��\CT����=q��p�@   Ah(�CV��                                    Bx�^/R  �          A ����(���G�@+�A�ffCT\)��(���(�@�Apz�CV}q                                    Bx�^=�  �          @��R���
���
?���Ap�CS�H���
����?@  @�{CT�\                                    Bx�^L�  �          A�H�������?���A#\)CTxR������\)?O\)@�
=CUp�                                    Bx�^[D  �          A  ��  ��G�?���ACU���  ��{?5@�CV��                                    Bx�^i�  �          A�
��\)��
=?��A733CU�\��\)���?xQ�@�G�CV��                                    Bx�^x�  �          A
=�������?�33AW\)CS����������?��
A(�CT�3                                    Bx�^�6  �          A ����p����\?�A\z�CS}q��p���=q?�ffA�CT�                                    Bx�^��  �          A���Q����\?�{AT��CS
��Q�����?�  ACT}q                                    Bx�^��  �          A�H��G���?�ffAK�CS����G�����?�A�CT�R                                    Bx�^�(  �          A\)�����?�AX��CU������H?�G�Ap�CV}q                                    Bx�^��  �          A33�������\@33Ag�
CS�������H?�A Q�CT�
                                    Bx�^�t  �          A��ڏ\���?�33AQ�CT���ڏ\����?=p�@�(�CUu�                                    Bx�^�  �          A�R��Q����?�
=A>�RCWB���Q�����?z�H@�p�CXh�                                    Bx�^��  �          A�\��������@ ��Ac�
CW�q������z�?��A�CYh�                                    Bx�^�f  �          A�
��������@��Ax(�CV����������?�  A((�CXT{                                    Bx�_  �          Aff�У���  ?�z�AYp�CV�f�У����?�(�A	p�CX�                                    Bx�_�  �          @�����H��=q?�=q@�=qCUE���H��>�
=@AG�CU��                                    Bx�_(X  �          @�
=��p��xQ�>\)?���CR5���p��w
=��G��S33CR�                                    Bx�_6�  �          @���\�r�\���p(�CS���\�]p��{���CQ�f                                    Bx�_E�  �          @��
�ə�����p����\CUp��ə��~{�Ǯ�=�CT5�                                    Bx�_TJ  �          @��H�Ϯ���=�\)>�CU�)�Ϯ��=q�������CU�)                                    Bx�_b�  �          @�33��G���{�#�
����CX�R��G���(��B�\����CX\)                                    Bx�_q�  �          @�����
=���<��
=�CU�3��
=��녿&ff����CU�f                                    Bx�_�<  �          @�33��=q����>#�
?�CU���=q��Q�   �j�HCT�                                    Bx�_��  �          @����
���\>W
=?�G�CU.���
��녾��W
=CU\                                    Bx�_��  �          @�������(��������CXG����������\���CWn                                    Bx�_�.  �          A ���љ����;Ǯ�333CWc��љ����ÿ�{���RCV��                                    Bx�_��  �          A ����Q������
=��CTz���Q������  �\)CS��                                    Bx�_�z  �          @�����
�k�?�  @���CP0����
�r�\>��@=p�CP��                                    Bx�_�   �          @������Vff?(�@��CNL������Y��=�\)?\)CN��                                    Bx�_��  
�          @��R��p��z=q�L�Ϳ��HCQz���p��u��Q����CP�R                                    Bx�_�l  �          A �����H�o\)�Ǯ�0  COٚ���H�h�ÿxQ���\)CO(�                                    Bx�`  �          A�R���
�~{��p��%CQ����
�w
=��  ��Q�CPp�                                    Bx�`�  �          A����(�����c�
��z�CRB���(��}p�����+33CQ�                                    Bx�`!^  �          A�\�����\)���\�߮CT�������Q��p��=�CSB�                                    Bx�`0  �          AQ���\)���
��\)��CS+���\)��(����D  CQ�                                    Bx�`>�  �          Az�������׿n{�ȣ�CRE�����녿�{�.{CQ�                                    Bx�`MP  �          A(�����Q�>L��?���CR{��������H�Tz�CQ�3                                    Bx�`[�  �          AQ������
=    ���
CR���������:�H����CRc�                                    Bx�`j�  �          A
ff��  ��  >��?xQ�CQ����  ���R����i��CQW
                                    Bx�`yB  �          A	G��������>���@   CP�f��������\�!G�CP�)                                    Bx�`��  �          A����  ��(�>aG�?�(�CP޸��  ������ECP                                    Bx�`��  �          A	p���
=��\)>�@Dz�CQ����
=����u��\)CQ��                                    Bx�`�4  
�          A
{��\)��Q�?!G�@��RCQ�f��\)�������
��CQ�                                    Bx�`��  �          A
�\��\)��G�?G�@�{CQٚ��\)���=u>ǮCRB�                                    Bx�`  �          A\)��\��ff?c�
@�33CP�q��\��G�>8Q�?�z�CQ�                                     Bx�`�&  �          A\)��(���=q?��\@�\)CP���(���{>���@��CP                                    Bx�`��  �          A
�H���{�?}p�@ҏ\CO�����G�>��
@
=CO�                                    Bx�`�r  �          A  ���R�~�R?�z�@���COL����R���
>�@HQ�CP)                                    Bx�`�  �          A����z�H?��\A�CO\�����\?
=@w�CO�q                                    Bx�a�  �          A  ��
=�w�?�z�A��CN����
=����?=p�@�=qCO�                                    Bx�ad  �          A���ff�u�?�(�A�CNxR��ff����?L��@�  CO��                                    Bx�a)
  �          Az�����_\)?�ff@��
CK������g
=>�G�@7
=CLW
                                    Bx�a7�  �          A
=��ff�a�?�(�AQG�CL�)��ff�s�
?�=qAG�CNW
                                    Bx�aFV  �          A
�R��p��^{@z�A]G�CL^���p��q�?�Q�AG�CN:�                                    Bx�aT�  �          A
ff��p��Z=q@A_�CK����p��n{?�(�AQ�CM�H                                    Bx�ac�  T          A
�H��(��\(�@33AuCLE��(��r�\?�A1G�CNff                                    Bx�arH  �          A
=q��
=�P��@�A^�HCJ�f��
=�dz�?�p�A�CL�
                                    Bx�a��  �          A	���z��R�\@  AqCKO\��z��hQ�?��A/\)CMp�                                    Bx�a��  �          A	����\)�Y��@#33A�CLu���\)�r�\?�z�AMp�CN�f                                    Bx�a�:  �          A	p���ff�U@*�HA��HCL(���ff�p��@�\A\  CNǮ                                    Bx�a��  �          A33���\�G
=@
=qAe��CI�����\�\(�?���A&�RCK�                                     Bx�a��  �          A
ff��(��J=q@#33A�
=CJ}q��(��c�
?���APQ�CM�                                    Bx�a�,  �          A
{���I��@:=qA���CJǮ���g
=@�
Aw�
CM�                                     Bx�a��  �          A
�\��R�R�\@;�A��CK�=��R�p��@�\Aup�CN�q                                    Bx�a�x  �          A
=q��(��Q�@C�
A��CK����(��q�@�HA�Q�CO�                                    Bx�a�  �          A
{��{�L(�@?\)A�=qCK5���{�j�H@�A
=CNJ=                                    Bx�b�  �          A
{��(��P��@C33A��CK����(��p��@=qA��CN��                                    Bx�bj  �          A
�\��{�N�R@A�A��CKz���{�n�R@��A��\CN�)                                    Bx�b"  �          A���(��4z�@G�A��CHO\��(��U@#�
A�z�CK��                                    Bx�b0�  �          A33����8Q�@L��A�ffCH�
����Z=q@'�A�{CLJ=                                    Bx�b?\  �          A
�\���
�#33@N{A��HCFxR���
�E@-p�A���CJ{                                    Bx�bN  �          A
�R��33�%�@P  A�ffCF� ��33�HQ�@.{A���CJff                                    Bx�b\�  �          A
�R��33�\)@Tz�A�Q�CF���33�C�
@3�
A�33CI�f                                    Bx�bkN  �          A
�H��p��\)@XQ�A��CDL���p��5�@9��A�ffCH:�                                    Bx�by�  T          A
�\��p����R@^�RA�CB����p��&ff@C33A�G�CF�3                                    Bx�b��  �          A	���=q��p�@`  A���CB�f��=q�%@Dz�A�(�CF�                                    Bx�b�@  T          A
=���Ϳ�ff@w�A�(�C=�f�����   @c�
A�  CB��                                    Bx�b��  �          A
=�������@}p�A�33C;�������@l(�A�\)CA.                                    Bx�b��  �          A\)���׿L��@uAѮC9�f���׿�G�@g�A��HC?�                                    Bx�b�2  T          A	���Q�n{@�Q�A�G�C;\��Q��@p��A�ffC@�=                                    Bx�b��  �          A=q��\��33@xQ�Aܣ�C<�3��\��\)@eA���CBJ=                                    Bx�b�~  �          A
=��33��{@s�
A�Q�C>޸��33��
@^�RA�CD33                                    Bx�b�$  �          A������Q�@mp�A�z�CD�������333@O\)A�G�CI�q                                    Bx�b��  �          Az���Q��ff@p��AمCD���Q��1�@R�\A�z�CI��                                    Bx�cp  �          A�
��G�����?�
=A<��Cg@ ��G���\)>�ff@G
=Ch&f                                    Bx�c  �          A�\��Q���{?��
A,��Ci����Q��Ӆ>��?�=qCj=q                                    Bx�c)�  T          A{����Ӆ?ǮA+�Cih������G�>�  ?�(�Cj!H                                    Bx�c8b  �          A���
=��(�?�G�A?�Ci)��
=���H>�G�@>{Ci�q                                    Bx�cG  �          A\)�������?�33A��Cj�{�����޸R=L��>��
Ckc�                                    Bx�cU�  T          A�
��ff���H?��\A	p�Cg�q��ff�ָR�#�
����Ch@                                     Bx�cdT  
�          A
=��Q����
?�33@��CkL���Q��޸R�aG��\Ck�f                                    Bx�cr�  b          A
=��p��޸R?u@�Q�Cl(���p���Q��(��<(�ClY�                                    Bx�c��  �          A=q��p����?z�H@�{Cm���p�����G��>�RCn�                                    Bx�c�F  T          A{��\)���?aG�@���Cms3��\)��녿��hQ�Cm�{                                    Bx�c��  �          A�H������G�?h��@ƸRCm{������\��\�\(�Cm:�                                    Bx�c��  �          A(���z���33?J=q@�\)Co��z����#�
��\)Co�\                                    Bx�c�8  �          A
=�z�H��?�G�@��Cq��z�H��p���G��C�
CqL�                                    Bx�c��  �          A�R��p���{?k�@�\)Cn����p���\)���H�^{Co!H                                    Bx�cل  �          A���=q��=q?B�\@�(�Cm�f��=q���H�!G����RCm�3                                    Bx�c�*  �          A{���R��>�(�@@��Cn�����R���
�z�H��
=Cn�                                    Bx�c��  �          Aff��z���=q?.{@��Cm@ ��z���=q�8Q�����Cm=q                                    Bx�dv  �          AG����H��Q�?E�@�
=CmJ=���H�أ׿�R��CmW
                                    Bx�d  �          A�H��p��޸R?u@׮Co
��p���Q���Y��CoB�                                    Bx�d"�  �          A���������>��@7�Cp�f������\�����{Cpk�                                    Bx�d1h  
�          AQ����H��  ?xQ�@أ�Ck����H�ٙ���G��Dz�Ck�q                                    Bx�d@  �          A��������p�?fff@�  Cjp�������
=�   �]p�Cj�)                                    Bx�dN�  �          A�������?��RA\  Ch������p�?�R@��Ci8R                                    Bx�d]Z  �          A���\�׮?5@�z�Cjff���\�׮�5��(�Cjff                                    Bx�dl   �          A�H���\�޸R?^�R@�{Cl�)���\��\)�����(�Cl��                                    Bx�dz�  �          A�H���
��G�?�G�A%��Cn.���
��ff<#�
=#�
Cn                                    Bx�d�L  �          A
=�w
=��G�?��
AB�\Cr��w
=��  >W
=?�33Cr�                                    Bx�d��  �          A  ������
?s33@�ffCq&f��������R��
=CqB�                                    Bx�d��  �          Az����H��?h��@�33Cq+����H��ff�.{��33Cq@                                     Bx�d�>  �          A����\)��33?��
@�ffCp\��\)���Ϳ���k�Cp=q                                    Bx�d��  �          A
{��=q���
?���@�z�Co����=q�����^�RCo��                                    Bx�dҊ  �          A	��x�����
?fff@�G�Cr���x����z�=p����RCs�                                    Bx�d�0  �          A	����\)>��?�G�C}�����G���  �!C}@                                     Bx�d��  �          A�������
=>�?fffC|�f�����녿˅�/\)C|�{                                    Bx�d�|  �          A
=��  ��z�?�p�A?
=Cp����  ���H>��?��
Cqff                                    Bx�e"  �          A���33��z�@
�HAnffCm����33��ff?!G�@�Q�Cn�{                                    Bx�e�  �          A	���G���\)@@  A��RCa�R��G���  ?�
=A5�Cdk�                                    Bx�e*n  "          A{���H��@,(�A�33Cg����H���
?��
A��Ci��                                    Bx�e9  �          A  ��=q��{@\)A��HCk����=q��\?xQ�@ҏ\Cm(�                                    Bx�eG�  �          Az�������H@ ��AZ�\Ck�R�����>�@G
=Cl                                    Bx�eV`  �          A  ��z���@%A�  Ch����z���33?�{@�Q�Cjn                                    Bx�ee  �          A���\)��p�@FffA��
Ca���\)�ƸR?�\AAp�Cd�\                                    Bx�es�  �          Aff��p���\)@W�A��RC^�f��p����H@Q�Aj{Ca�
                                    Bx�e�R  �          AG���  ���H@G
=A�Q�C]}q��  ����?�33AT(�C`}q                                    Bx�e��  T          A
=���R��=q@+�A�{C]�R���R����?�A\)C`�                                    Bx�e��  �          A����R���H@l��A�  CZ�\���R����@$z�A�(�C^�H                                    Bx�e�D  	�          A  ��\)��p�@Q�A��C\����\)����@Ak�C`                                      Bx�e��  
�          Ap��Å��z�@6ffA�(�CZ�H�Å����?�
=A9�C]u�                                    Bx�eː  �          AG������R@,(�A��HC_ٚ����p�?��A��Cb33                                    Bx�e�6  �          A�����
��@1G�A�=qCb�����
���?�z�A��Cd��                                    Bx�e��  T          A�H������@ ��A��HCd33������?��@���Cf
                                    Bx�e��  
(          A33��p�����@Dz�A�(�CgY���p���{?�=qA,��Ci��                                    Bx�f(  �          A33����@?\)A�  Cgh�����ff?�  A#�Ci�f                                    Bx�f�  T          A�\������z�@(Q�A�{CiB�������=q?��@�\)Ck�                                    Bx�f#t  
�          A�\��z����
@I��A�(�Cb8R��z���ff?�\AB�\Cd�q                                    Bx�f2  �          A�H��
=���R@|(�A��C[s3��
=���@/\)A��RC_��                                    Bx�f@�  �          A33��33��Q�@�=qA���CY����33���\@:=qA��C^Q�                                    Bx�fOf  T          A�R����n�R@�33A�33CS�������z�@W
=A��CYn                                    Bx�f^  �          A�����\����@{�A��C\z����\����@-p�A�C`�{                                    Bx�fl�  �          A�
�Ϯ�1G�@���A�CK#��Ϯ�j�H@X��A�p�CQxR                                    Bx�f{X  �          A�H�ʏ\�fff@n{A�z�CQ���ʏ\��(�@1G�A�ffCV�                                    Bx�f��  �          A������{�@�(�A��
CU��������G�@EA�
=C[\                                    Bx�f��  �          A����\��33@vffA�p�CX�����\��(�@.{A�Q�C]^�                                    Bx�f�J  �          A������z�@fffA͙�CZ\)������@��A�Q�C^p�                                    Bx�f��  "          A�����H����@i��A���CW
���H����@!�A�  C[��                                    Bx�fĖ  "          AQ��������H@eA�{CW�H������=q@��A�=qC[�3                                    Bx�f�<  
�          A�R�����(�@P��A�ffCW޸�����G�@�Ap��C[�R                                    Bx�f��  �          Ap�����Q�@u�A�=qCP������33@;�A�
=CU�f                                    Bx�f��  �          AG�����C33@���A�33CM�=����{�@K�A��CS�                                    Bx�f�.  �          A (���=q�Fff@tz�A�CN+���=q�{�@<��A��CS��                                    Bx�g�  �          @�
=����@  @k�A�CM\����s33@5A��CR��                                    Bx�gz  �          @��R��33�Dz�@k�A܏\CM�{��33�xQ�@5�A�ffCSff                                    Bx�g+   b          @��R���H�P��@b�\Aә�CO8R���H����@(��A�33CTk�                                    Bx�g9�  x          @�Q��ٙ���p�@c33A�=qC7��ٙ���p�@VffA���C>:�                                    Bx�gHl  "          @�Q�����+�@n�RA�ffC9�3����Ǯ@\��A��HCA��                                    Bx�gW  �          @��H��\)�;�@{�A�(�CN
��\)�s�
@EA�(�CTu�                                    Bx�ge�  "          @������R����@3�
A���C]T{���R��33?��
A5��C`ff                                    Bx�gt^  �          @�����������@P��A�  CV��������ff@
=qA��HC[�                                    Bx�g�  �          @����=q�`  @^{A���CQ����=q��  @\)A��RCW�                                    Bx�g��  �          @�Q���p��B�\@G�A��
CML���p��n{@G�A��CR�                                    Bx�g�P  �          @��R�˅�Dz�@C�
A��CM��˅�o\)@(�A�ffCRn                                    Bx�g��  b          A   ������Q�@33A��CX��������{?���A (�C[\)                                    Bx�g��  �          @�G���Q����
@
=AzffCa����Q���
=?0��@�=qCch�                                    Bx�g�B  T          A���(�����@G�A�Q�C]�R��(���{?^�R@���C`\                                    Bx�g��  �          A  ��  �k�@?\)A��RCQ� ��  ��=q?�(�Ab=qCU�{                                    Bx�g�  �          A33��  ?^�R@��B��C,����  ��(�@���B�C7��                                    Bx�g�4  �          A	����#�
@��B�HC4\)������@�B  C>��                                    Bx�h�  �          A	���33�Y��@���B  C;���33���@���B\)CEY�                                    Bx�h�  b          A��أ׿�  @�{B��C>}q�أ���R@�ffA��CH{                                    Bx�h$&  �          A����G��G�@�z�A��
CMz���G���z�@\��A�  CTT{                                    Bx�h2�  �          A�H�����1�@��
A�{CJ�������s�
@`��A���CQ��                                    Bx�hAr  b          A��
=�=q@��\A�CG�R��
=�\��@c�
A��
CO!H                                    Bx�hP            A�H�����8Q�@qG�A�CKh������p��@9��A�{CQu�                                    Bx�h^�  �          A�������8��@tz�A�
=CK�
�����q�@<(�A�(�CR�                                    Bx�hmd  �          A��G��b�\@k�A�  CQaH��G���(�@(Q�A��CV�
                                    Bx�h|
  T          A Q��������@S33A\CU�������@�Av=qCZ5�                                    Bx�h��  �          @�{��(����R@K�A��
CW�\��(���z�?�(�Af�RC[�q                                    Bx�h�V  �          A�\��33�Mp�@\)A�{CN�=��33��z�@@  A��HCU)                                    Bx�h��  �          A���33�Y��@c33Ạ�COB���33��
=@!�A�\)CT�{                                    Bx�h��  �          A����#�
@n{AծCH=q���\��@:=qA���CNp�                                    Bx�h�H  �          A33��(���@|��A�
=C=��(���@`  AîCDp�                                    Bx�h��  �          A�H��{�0��@��\A�G�CJs3��{�s�
@\(�A��RCQ��                                    Bx�h�  �          A�
�θR�fff@�A��
CQ&f�θR���\@Dz�A��CW^�                                    Bx�h�:  T          A(��ҏ\�Vff@�\)A�\CO  �ҏ\���@K�A��CU�=                                    Bx�h��  "          A\)��G��\��@��A�Q�COǮ��G���p�@A�A�{CV�                                    Bx�i�  
�          A33���c33@�p�A�z�CP������G�@C�
A��CW5�                                    Bx�i,  �          A33�ə��l(�@��A�33CR\)�ə���ff@EA��CX�3                                    Bx�i+�  T          A
=��ff����@���A�33CU+���ff��  @333A�
=CZٚ                                    Bx�i:x  b          A�R��G��w
=@�=qA���CS�\��G���=q@7�A�G�CYz�                                    Bx�iI  x          AQ���  �Dz�@�p�B�CN+���  ���@Z�HA�CU��                                    Bx�iW�  
�          A33��ff�H��@�33B G�CNٚ��ff��ff@U�A��CV&f                                    Bx�ifj  "          A�H�����HQ�@���B
=CN�3�������R@XQ�A���CVc�                                    Bx�iu  
�          A��ə��8Q�@�{B��CL�\�ə��\)@^�RA�=qCTO\                                    Bx�i��  
�          A\)��Q��Fff@��A���CNQ���Q����@S33A�p�CU�H                                    Bx�i�\  
�          A  ����I��@���A�{CN��������R@P  A�p�CU�3                                    Bx�i�  
�          A��������(�@l(�A�  CU��������@=qA�G�C[
                                    Bx�i��  
Z          A33��  �1G�@�  BQ�CK�f��  �z=q@c33A��CT�                                    Bx�i�N  �          A�H��{�#�
@�{B�CJz���{�p��@s33A�\)CSO\                                    Bx�i��  "          A
=���*=q@���B
�CKE���vff@n{Aأ�CS��                                    Bx�iۚ  
�          A �������7�@�\)B�CMxR������Q�@`  AΣ�CU�H                                    Bx�i�@            A (���  �@  @�(�B��CN����  ���@W
=AƸRCVc�                                    Bx�i��  �          @�z����
�A�@���B(�COE���
���@Q�A�ffCW                                      Bx�j�  �          @��
��\)�HQ�@�(�Bp�CP����\)���@Tz�A��CX�                                     Bx�j2  �          @��
��(��Q�@���B
=CR:���(���z�@R�\A�{CY�                                    Bx�j$�  T          @�z����H�;�@�p�B=qCN�)���H���@Z=qA���CVǮ                                    Bx�j3~  �          @��\��Q��/\)@���B  CMu���Q��z�H@eA��
CV@                                     Bx�jB$  �          @��������,��@�B
=CM\�����vff@^�RAԸRCU��                                    Bx�jP�  �          @�����Q��.�R@�ffB�RCMW
��Q��x��@_\)A�G�CV                                      Bx�j_p  
�          @����=q�0��@�G�B�\CMc���=q�w�@Tz�A��CU��                                    Bx�jn  
�          @�G�����*�H@�=qB��CL\)����s33@XQ�A�\)CT�                                    Bx�j|�  &          @�
=��33�,(�@�  B�CL���33�s33@S33A�ffCT��                                    Bx�j�b  �          @�  ��{�"�\@�Q�B��CK)��{�j=q@VffA��CS�)                                    Bx�j�  �          @��R��=q�$z�@�33B	��CKٚ��=q�n{@[�A��CT��                                    Bx�j��  �          @�
=��G��0  @�G�B=qCMs3��G��xQ�@S�
A�G�CU�
                                    Bx�j�T  �          @��
����@��@���B(�CPh�������
@Mp�A�33CX��                                    Bx�j��  �          @��
��G��S33@�G�B =qCRǮ��G����\@9��A�p�CZ�                                    Bx�jԠ  �          @�\)��
=��33@�G�B�CF\)��
=�N{@���A�ffCQT{                                    Bx�j�F  �          @����(���\)@�G�BQ�CE�f��(��L(�@���A��RCP}q                                    Bx�j��  �          @�����33�\)@��HB�CH����33�_\)@n�RA�CR�                                    Bx�k �  �          @�{��z��p�@��\B��CK�)��z��l��@i��A�p�CUJ=                                    Bx�k8  �          @������\�#�
@��HB
=CM�����\�w�@w�A�33CW�3                                    Bx�k�  �          @�z������R@���B�RCM\����tz�@|��A��CW�                                     Bx�k,�  �          @�(������Q�@��Bp�CI������]p�@}p�A�z�CT�                                    Bx�k;*  T          @��
���
�5�@xQ�A��CM���
�w
=@7
=A�33CUL�                                    Bx�kI�  �          @��
��\)����@���Bz�CG�)��\)�Q�@~{A��CR�3                                    Bx�kXv  
�          @�z������R@���B\)CM{����s�
@uA�CW�H                                    Bx�kg  �          @�������
@�=qB��CH!H����Y��@~�RA�{CS@                                     Bx�ku�  �          @��R��p���\)@�z�B'p�CG��p��U�@��HB	G�CS�                                    Bx�k�h  �          @�ff�����	��@�z�BCI(������aG�@���A��CTc�                                    Bx�k�  �          @�����
��33@�Q�B-�HCB�{���
�:�H@��HB�RCP�                                    Bx�k��  �          @�p���33����@�=qB/��CB5���33�9��@��B��CPu�                                    Bx�k�Z  �          @�p�����G�@�ffB*�HCC�����@��@�  BCQ
=                                    Bx�k�   �          @�����=q��z�@��B,��CEW
��=q�J�H@�\)B�\CR��                                    Bx�kͦ  �          @��
��Q���@�ffB,ffCF�q��Q��R�\@��B��CS�R                                    Bx�k�L  �          @�\���
��@�p�B"{CI&f���
�^�R@���BffCT�3                                    Bx�k��  �          @�{���H�z�@��B�CHQ����H�]p�@\)A�{CS��                                    Bx�k��  �          @������\��@��B33CH����\�Z=q@}p�A��CSp�                                    Bx�l>  �          @�R���׿���@�
=B&�
CG
�����P��@�p�B�CS�                                    Bx�l�  �          @���p�����@��B3\)CC\��p��4z�@��B�CQ�{                                    Bx�l%�  �          @�=q���
���@�ffB1�HCC\)���
�333@���BCQ�                                    Bx�l40  �          @�=q����  @��HB,�RCD�)���:�H@�z�BG�CR�)                                    Bx�lB�  �          @�=q����  @��\B,�CD�����:�H@�(�B�CR��                                    Bx�lQ|  �          @�=q���H���@�p�B0G�CE�)���H�?\)@�ffB��CS�3                                    Bx�l`"  �          @��
�������@�=qB�CGff����I��@q�B �CSn                                    Bx�ln�  �          @�(���
=���R@�B��CH�
��
=�QG�@e�A�CT�                                    Bx�l}n  �          @�����\)�G�@�BQ�CI\��\)�S33@e�A�Q�CT@                                     Bx�l�  �          @����G�@���B�CIO\���U@j=qA��CT��                                    Bx�l��  �          @��H�����@���B#��CG\����N�R@\)BCS�f                                    Bx�l�`  �          @����z��:�H@y��B��CPu���z���  @1�A��CX��                                    Bx�l�  �          @������/\)@��
B
Q�COJ=����y��@B�\A�=qCXL�                                    Bx�lƬ  �          @�����p��
=@�(�B�\CJ5���p��^{@n{A��HCUٚ                                    Bx�l�R  �          @�33��Q��p�@�=qBG�CI�
��Q��^{@Y��A܏\CT.                                    Bx�l��  �          @�\���H�*�H@z�HBG�CM�����H�q�@7
=A�{CV�                                    Bx�l�  �          @�����R�.�R@vffA�ffCM�����R�s�
@1�A�=qCU�R                                    Bx�mD  �          @������8��@�33B=qCO{�������@=p�A�=qCW��                                    Bx�m�  �          @�ff��ff�<(�@W�A�33CNQ���ff�w�@  A���CU                                    Bx�m�  �          @�����H�>�R@<��A�CN����H�qG�?�=qAe��CS��                                    Bx�m-6  �          @�\)�����/\)@�G�B(�CM޸�����x��@<��A�  CV�{                                    Bx�m;�  �          @�ff����6ff@r�\A�\CNk�����z�H@*�HA��
CVT{                                    Bx�mJ�  �          @�33��33�4z�@uA���CN� ��33�z=q@.{A���CV�                                    Bx�mY(  T          @�
=����1G�@|(�B��COL�����x��@5�A��CW�q                                    Bx�mg�  �          @�
=����R�\@FffA��
CRn������
?�{Ao�
CXQ�                                    Bx�mvt  T          @�p�����`��@\)A��CU�������(�?�(�A$  CY��                                    Bx�m�  �          @޸R���\�C�
@3�
A���CP�3���\�s�
?�A]��CVJ=                                    Bx�m��  �          @���
=�Fff@8Q�A��HCQ�\��
=�w�?��HAd��CWL�                                    Bx�m�f  �          @�Q����H�B�\@X��A�G�CQ�����H�~�R@p�A��HCX�q                                    Bx�m�  �          @�ff��\)�5@eA��\CP� ��\)�w
=@p�A��CXk�                                    Bx�m��  �          @��H����@  @S33A�CR+�����z�H@Q�A��CY5�                                    Bx�m�X  �          @�z���
=�I��@/\)A��CQ����
=�xQ�?�ffAP  CWY�                                    Bx�m��  �          @��
�����Z�H@\)A�=qCS������}p�?}p�A(�CW��                                    Bx�m�  �          @�G����
�Fff@333A£�CQ�q���
�vff?�\)A\z�CW�f                                    Bx�m�J  �          @ڏ\��(��>�R@:�HA���CP����(��q�?�\Apz�CW�                                    Bx�n�  �          @�{��G��[�@9��A�G�CU  ��G���ff?˅AT  CZu�                                    Bx�n�  �          @������^�R@7
=A�  CT�������\)?��AJ�RCZ�                                    Bx�n&<  �          @�(���Q��U@C�
A��CS=q��Q���p�?�\AeCY�                                    Bx�n4�  
�          @�z���
=�S�
@I��A�(�CS0���
=��p�?�\)As
=CY\)                                    Bx�nC�  �          @�(���  �fff@0��A�(�CU0���  ���?�33A4��CZ�                                    Bx�nR.  �          @��
����j=q@   A���CUc������G�?���AffCY��                                    Bx�n`�  �          @����(��<(�@O\)A���CP�f��(��w
=@�
A�z�CW��                                    Bx�noz  �          @�\)��Q��	��@��
BCK5���Q��_\)@X��A�(�CV��                                    Bx�n~   �          @�{��ff�'�@qG�BCN���ff�o\)@*�HA��HCW�q                                    Bx�n��  �          @����
=�333@UA�p�CP+���
=�p��@��A���CW�                                    Bx�n�l  �          @��������%�@mp�Bp�CO.�����k�@'�A��RCXE                                    Bx�n�  �          @ٙ����\��
@�B�HCM�����\�e@I��A��HCX��                                    Bx�n��  �          @���  ��@tz�BQ�CLL���  �a�@2�\A�ffCU��                                    Bx�n�^  �          @������H�'�@s�
B��CO8R���H�p��@,(�A�ffCXs3                                    Bx�n�  �          @�=q�����!G�@u�B��CN�������k�@/\)A��CX&f                                    Bx�n�  �          @��H�����$z�@u�B�
CO������n{@.{A�ffCX��                                    Bx�n�P  �          @�����\)�2�\@P��A�z�CP{��\)�o\)@
=A�33CW�\                                    Bx�o�  �          @ҏ\�����<(�@B�\AܸRCR\)�����s33?�A���CY(�                                    Bx�o�  �          @��H��  �)��@W�A��\CO����  �i��@��A�z�CX�                                    Bx�oB  �          @Ӆ��{�0��@A�A�z�CP��{�h��?�33A�\)CW
=                                    Bx�o-�  �          @�Q���Q��7
=@#�
A��CP���Q��dz�?�AIp�CV&f                                    Bx�o<�  �          @θR����5@\)A�
=CPs3����a�?���A@��CU�                                    Bx�oK4  �          @���ff�3�
@�HA�(�CPh���ff�^�R?�ffA;\)CU�                                    Bx�oY�  �          @�G���  ��@c33B=qCL�=��  �W�@!�A�(�CU�q                                    Bx�oh�  �          @Ϯ����R@eBQ�CLff���U@%A�{CV+�                                    Bx�ow&  �          @�z���\)�z�@N{A��CL���\)�R�\@p�A�=qCUp�                                    Bx�o��  �          @θR����p�@J�HA�\)CK33����K�@��A�G�CS�H                                    Bx�o�r  �          @�ff������R@I��A�z�CKp������L��@
�HA�CS��                                    Bx�o�  b          @���Q��Q�@I��A�z�CH@ ��Q��:�H@G�A�G�CQ�                                    Bx�o��  �          @�������ff@W�A�ffCJ�������I��@�HA��CT
                                    Bx�o�d  �          @���������@QG�A�=qCK\)�����L(�@33A�ffCTO\                                    Bx�o�
  �          @��
��G���\@I��A���CLn��G��P  @��A�
=CT�{                                    Bx�oݰ  �          @��
��{��@1G�A�(�CM���{�P  ?�p�A|��CT
=                                    Bx�o�V  �          @ʏ\���
��R@\)A�=qCL�����
�G
=?�(�A2=qCR)                                    Bx�o��  �          @��
��
=�"�\@ ��A�\)CL���
=�E?z�HA��CQs3                                    Bx�p	�  �          @θR��G��  @P��A�z�CL���G��P��@  A�Q�CT��                                    Bx�pH  T          @У����R��@e�B{CL�f���R�Y��@"�\A��CVn                                    Bx�p&�  �          @�Q���z����@dz�B
=CN&f��z��`��@�RA�\)CW��                                    Bx�p5�  �          @��
��
=��
@XQ�B �CJ�=��
=�HQ�@�HA��CT:�                                    Bx�pD:  �          @����{��
=@O\)A��CHc���{�=p�@A���CQ��                                    Bx�pR�  T          @�{��Q���R@>�RA�Q�CJ����Q��I��?�p�A��CR�f                                    Bx�pa�  �          @�33���R���@5�A�CKs3���R�G�?�=qA�(�CR�                                    Bx�pp,  �          @�33��ff���R@C33A�CH����ff�<��@Q�A���CQ��                                    Bx�p~�  �          @�������
=@,(�AƸRCIW
����<(�?�  A}CP��                                    Bx�p�x  �          @�{��\)���@�A�
=CK����\)�E?���AACQc�                                    Bx�p�  �          @˅���H�$z�@�A�  CM�����H�N{?�Q�A-p�CS{                                    Bx�p��  �          @��H��{�G�@>{A�Q�CI@ ��{�=p�@�\A���CQ��                                    Bx�p�j  �          @�=q��33�HQ�@,��A�z�CS�=��33�y��?��AC33CYaH                                    Bx�p�  �          @����33�B�\@+�A��CR�=��33�s33?��AEG�CX�3                                    Bx�pֶ  �          @��H��p��R�\@8Q�Aϙ�CU�=��p����
?��RAQ�C[�f                                    Bx�p�\  �          @У���G��:�H@��A���CP�f��G��e?�
=A&�HCV0�                                    Bx�p�  �          @У������>�R@��A���CQn�����i��?�33A"�RCV��                                    Bx�q�  T          @��H��
=�Q�@��A�  CUxR��
=�u?J=q@�ffCY��                                    Bx�qN  �          @�ff����>{?�{A��HCRp�����\��?&ff@�G�CVE                                    Bx�q�  �          @�z�����Z=q?�33ARffCV�
����l(�=L��>�CX�                                    Bx�q.�  �          @�33��ff�J=q?�{Au�CT����ff�a�>�{@L(�CW}q                                    Bx�q=@  �          @�ff��ff�E@�\A�  CS����ff�g�?B�\@�=qCX33                                    Bx�qK�  �          @����G��b�\?���Ak�
CXp���G��w
=>#�
?��HCZ�f                                    Bx�qZ�  �          @�\)��
=�P��?�
=A��CUG���
=�o\)?
=@���CX��                                    Bx�qi2  �          @��H���
�&ff@�A��CN����
�J=q?n{A(�CS��                                    Bx�qw�  �          @�����\�0  ?�=qA�ffCPz����\�N�R?.{@�{CT�                                     Bx�q�~  �          @��R��Q��+�?���A�  CP!H��Q��J=q?333@�\)CT@                                     Bx�q�$  �          @�������Z=q?�\)A���CW�������w
=>�@�{CZ�                                    Bx�q��  �          @�p���\)�q�?�
=A���C\(���\)��\)>Ǯ@fffC_Q�                                    Bx�q�p  �          @��H��Q��l��?���A}�C]
=��Q�����=�G�?�ffC_u�                                    Bx�q�  T          @�z����R�tz�?\ApQ�C^:����R����#�
�uC`W
                                    Bx�qϼ  �          @�=q�tz���
=?�Q�A<��Cc�
�tz���33������
Cd�                                    Bx�q�b  �          @���vff��?L��A ��CcY��vff���Q���RCcT{                                    Bx�q�  �          @��
�j�H��?+�@�33Cd���j�H��(��p����HCdY�                                    Bx�q��  �          @�  �l����G�?z�HA��Ce5��l�����H�333��
=Ce��                                    Bx�r
T  �          @��\�]p���z�?�
=AmG�Cf{�]p����
�k���Cg��                                    Bx�r�  �          @���Q���G�?��HAz�RCf��Q���G��#�
�޸RCh��                                    Bx�r'�  �          @�Q��aG���  ?��Ai��Cd���aG���\)�aG���CfG�                                    Bx�r6F  �          @�  �e��{�?�
=Ap��Cc��e��������{Cen                                    Bx�rD�  �          @�\)�S33��33?��A�(�Cg��S33���
�����\Ch�                                    Bx�rS�  �          @���J�H��  ?�33Al(�CiG��J�H��
=���R�P  Cj��                                    Bx�rb8  �          @���Mp���33?�ffAW�
Ci�)�Mp����׾�G����\Cj��                                    Bx�rp�  �          @��H�C�
��\)?��A]p�Ck���C�
���;���
=Cl��                                    Bx�r�  
�          @��
�Mp�����?�\)Ab{Ci�3�Mp���33�������HCk�                                    Bx�r�*  �          @�Q��C�
��{?�\)A;�Ckn�C�
��G��&ff�أ�Ck��                                    Bx�r��  �          @����E����?�G�A+�
CjO\�E����5���Cj��                                    Bx�r�v  �          @�\)�Fff����?�  A)G�CjǮ�Fff��ff�=p����Ck
                                    Bx�r�  �          @���XQ�����?�  A%�Cg�{�XQ�����8Q���p�Ch0�                                    Bx�r��  �          @����QG�����?�AC
=Ch���QG���p��\)���\Cis3                                    Bx�r�h  �          @����Tz�����?���A<Q�ChY��Tz���������Ci�                                    Bx�r�  �          @�G��^{��
=?c�
AQ�Cf�{�^{����J=q�\)Cf��                                    Bx�r��  �          @����Z=q���?�33A@Q�Cf�H�Z=q���ÿ
=q���\Cgp�                                    Bx�sZ  �          @��R�Y����33?�
=Ah��Ch  �Y����녾�{�^{CiW
                                    Bx�s   �          @��R�_\)����?�z�Ae�Cf�\�_\)�����{�Y��Ch.                                    Bx�s �  �          @�ff�[���ff?z�HA
=Ch^��[���\)�L���p�Ch�
                                    Bx�s/L  b          @�\)�u��  ?�33Ab{Cb33�u��\)��  �"�\Cc�=                                    Bx�s=�  �          @��R�g����R?��AX��CeG��g����;Ǯ�x��Cf��                                    Bx�sL�  �          @�ff�`  ��G�?��AYp�Cf��`  ��
=��
=���Cg�3                                    Bx�s[>  �          @�\)����,��?��RA��CT�R����C�
>���@j=qCX^�                                    Bx�si�  �          @�p���
=�E�@33A��
C<k���
=��G�?˅A���CD
                                    Bx�sx�  �          @��������G�@  A��CH
�����\)?��Af�\COff                                    Bx�s�0  �          @��\��p��L(�?��A�z�CYff��p��e�>�=q@5�C\�
                                    Bx�s��  �          @�(������R@�
AɮC;:�������H?��A�
=CD                                    Bx�s�|  �          @�(����Ϳ�33@	��A�p�CJ5������%?��RARffCQ�                                    Bx�s�"  �          @����Q��_\)?�A��HC]
=��Q��~{>��@��C`��                                    Bx�s��  �          @�
=�����Q�@  A�{CE�������?�A\Q�CL�                                    Bx�s�n  �          @���G���?�A�CF�H��G��=q?��A%CL�=                                    Bx�s�  �          @�33��G��33?ٙ�A�Q�CL���G��1�?(��@ҏ\CP޸                                    Bx�s��  T          @��������\?˅Au�CH�����   ?(��@�33CL�\                                    Bx�s�`  �          @\�����<(�?��HA���CRff�����XQ�>�
=@|��CU�                                    Bx�t  �          @��H��Q��(�@�\A��CJ����Q��3�
?��\A
=CP&f                                    Bx�t�  �          @������
�ٙ�@(�A��CE�{���
�!G�?�=qAnffCM+�                                    Bx�t(R  �          @��������@
�HA�G�CM�������E?��A33CSxR                                    Bx�t6�  �          @\����'�@�A�\)COQ�����O\)?k�A\)CT�)                                    Bx�tE�  �          @�ff�����1�?�=qAQG�CQ������E�>�?�G�CS�\                                    Bx�tTD  �          @��H��Q�� ��@�RA�z�CO����Q��L(�?��A(Q�CU�{                                    Bx�tb�  �          @����Q��=q?�A�
=CH)��Q��ff?O\)Ap�CM{                                    Bx�tq�  �          @�����R�(�?�\)A��CK�=���R�)��?!G�@�(�CP�                                    Bx�t�6  �          @�����H��ff?�p�Av{CG�����H�\)?&ff@�p�CK�=                                    Bx�t��  T          @�33��33��?��A�{CK&f��33�'�?Q�A(�CPQ�                                    Bx�t��  �          @�
=��Q���H?��A_
=CF�)��Q��?�@��CJ��                                    Bx�t�(  �          @�����R�h��@"�\A��C>^����R��?�(�A��CH\)                                    Bx�t��  �          @����\)��(�@Q�AǅCA����\)�z�?ٙ�A�(�CJ�
                                    Bx�t�t  �          @�G�����޸R@�RA��HCH33����\)?�{Ac\)CO�3                                    Bx�t�  �          @��������\@
=A�CN�������B�\?�  AO�CU�H                                    Bx�t��  �          @����{���@(�A��C5Ǯ��{��G�?�Q�A��HC?�
                                    Bx�t�f  �          @�����z�+�@'
=A�G�C;
��z��z�@�A�33CE�                                    Bx�u  �          @�=q��(����@A�A�{C?�q��(��p�@A�  CKJ=                                    Bx�u�  �          @�33��=q��ff@�
A��\CBW
��=q�
=?���A�  CJ�)                                    Bx�u!X  �          @\���=L��?У�A��HC3���녿(�?\ArffC:B�                                    Bx�u/�  T          @\��p��#�
?��\AB{C4k���p����?�33A/
=C9L�                                    Bx�u>�  �          @����ff�Q�?��A3�
C<5���ff���H?8Q�@���C@�                                    Bx�uMJ  T          @������\�   @   A��HCH�\���\�(Q�?��
AG�CNL�                                    Bx�u[�  �          @������R��33@!�A�z�CE�����R�!�?�33A~=qCM�f                                    Bx�uj�  �          @�{��녿�(�@7�A�33CJL�����=p�?�=qA�  CS�)                                    Bx�uy<  �          @�p���
=����@(��A�(�CD@ ��
=���?�=qA�33CM��                                    Bx�u��  �          @�p���z���@'�Aԣ�CI
=��z��1�?��A�(�CQ�
                                    Bx�u��  
(          @�ff��=q�z�@%A�
=CM����=q�J�H?�Q�A`  CUY�                                    Bx�u�.  �          @�����=q�  @{A�  CK���=q�C33?���AMCS\                                    Bx�u��  	�          @�
=��(���
@E�A�\)CMJ=��(��Vff?��A�
=CVz�                                    Bx�u�z  
�          @ȣ������*�H@@  A�
=CQ������i��?�z�Av�\CYL�                                    Bx�u�   
�          @ʏ\��=q�.�R@UB �\CR�\��=q�vff?���A���C\�                                    Bx�u��  
�          @�=q��p���(�@C33A�
=CH�{��p��B�\?��RA��CRh�                                    Bx�u�l  
          @ə���ff���@@��A�(�CG����ff�<��?�p�A���CQ�=                                    Bx�u�  x          @���=q��@8��A�ffCH����=q�:�H?�{A�G�CQ��                                    Bx�v�  
�          @����33�,��@(Q�A�ffCQ���33�a�?�ffAD  CX
=                                    Bx�v^  T          @��
�����e@4z�A��C]�3������p�?�\)A*{Cc��                                    Bx�v)  T          @�G���(��+�@C33A���CSu���(��l(�?�Q�A��C\�                                    Bx�v7�  
�          @�����z��@[�B	\)CJz���z��I��@ffA�  CV+�                                    Bx�vFP  T          @�33��Q����@Q�A���CD����Q���?\Aj�HCLǮ                                    Bx�vT�  
�          @�(���ff��\)@�RA�p�CB&f��ff�
=q?�p�Ab�\CI��                                    Bx�vc�  T          @�ff���ÿ��R@	��A��\C@������� ��?��HA_33CG�q                                    Bx�vrB  
(          @��
���ÿ5@Dz�A��C;W
���ÿ�33@ ��A��
CG                                      Bx�v��  
�          @����33���@�
A�=qC>�3��33��z�?�A|z�CF��                                    Bx�v��  �          @����H�J=q@!G�A��\C<\���H��  ?�p�A�33CEk�                                    Bx�v�4  
�          @Å��(��B�\@p�A��
C;�3��(�����?ٙ�A��CC�
                                    Bx�v��  
�          @�Q���G��aG�?�  AAp�C<����G�����?G�@�Q�C@�{                                    Bx�v��  �          @�ff������?���A�ffC8������?�(�AZ�HC?5�                                    Bx�v�&  
(          @�{���H�J=q?�
=A��
C;k����H�\?�Q�AN�\CA�q                                    Bx�v��  	`          @θR��=q�k�@   A���C<����=q��?���AO33CCW
                                    Bx�v�r  	n          @�Q���33�:�H@
=qA���C:ٚ��33�Ǯ?�Amp�CBO\                                    Bx�v�  F          @�p����׿!G�@�A�ffC:����׿���?�Aq��CA�=                                    Bx�w�  �          @�����  �\@��A��HC7����  ��p�?�ffA��RC?�\                                    Bx�wd  �          @�p���
=�z�@33A�C9�
��
=��p�?���A�(�CA��                                    Bx�w"
  �          @�ff���;Ǯ@#33A���C7Ǯ���Ϳ�z�@
=qA�
=CAh�                                    Bx�w0�  �          @�=q��
=���@'�Aď\C8)��
=���H@{A���CBJ=                                    Bx�w?V  �          @ə���33?�@2�\AҸRC.����33�.{@0  A��C:��                                    Bx�wM�  �          @ʏ\���?@  @�A�=qC,������#�
@	��A�{C5�                                    Bx�w\�  �          @�(���33�8Q�@ffA�C;���33��\)?�A���CC��                                    Bx�wkH  �          @����=q=�G�@�A��C3  ��=q�=p�?��A��C:��                                    Bx�wy�  �          @�\)��z�?k�?���A��HC+u���z�=��
@�\A�
=C35�                                    Bx�w��  �          @θR��
=?��H?�p�A��\C&8R��
=?�\@��A���C/{                                    Bx�w�:  �          @������?�?޸RA|��C&�H���?�@
�HA�
=C.��                                    Bx�w��  �          @�z���{?��H?��RA���C(����{>�=q@�
A�{C1ff                                    Bx�w��  �          @���  ?��H?��A��C&^���  ?z�@�RA�ffC.u�                                    Bx�w�,  �          @�=q��  ?\)?�
=A�z�C.�
��  ��z�?��RA���C6�H                                    Bx�w��  �          @�G��ƸR?�ff?�z�AlQ�C*p��ƸR>��?�Q�A�
=C1�
                                    Bx�w�x  �          @Ϯ��{?n{?�p�AUG�C+xR��{>k�?�p�Az=qC1��                                    Bx�w�  �          @�{���
?�Q�?��AG
=C&� ���
?:�H?�{A�ffC-+�                                    Bx�w��  T          @�  ��=q?\?޸RAxQ�C%���=q?(��@p�A�p�C-��                                    Bx�xj  �          @�G���p�?!G�@�A��HC.+���p���=q@
=A���C6��                                    Bx�x  �          @����?��@�A��C.�����\@
=qA��C7�=                                    Bx�x)�  �          @Ϯ�\?L��@
=A��
C,}q�\���@  A��\C5k�                                    Bx�x8\  �          @Ϯ�\?\(�@33A�p�C+�3�\��\)@{A�Q�C4�                                    Bx�xG  �          @У���=q?��@33A��RC)Q���=q>8Q�@A�z�C2B�                                    Bx�xU�  �          @�  ����?��R@G�A���C(h�����>�z�@ffA��C1L�                                    Bx�xdN  �          @љ�����?��R?�(�A�ffC&(�����?�@=qA�\)C.�q                                    Bx�xr�  �          @�\)��Q�?^�R@�RA��C+��Q�\)@��A���C5^�                                    Bx�x��  �          @��H���H?�33@�A�
=C(ٚ���H>�@p�A��HC2��                                    Bx�x�@  �          @��H���
?˅?��
A�(�C$ٚ���
?333@�A�z�C-33                                    Bx�x��  �          @��
��Q�?޸R@z�A�{C#8R��Q�?5@'
=A�C,�q                                    Bx�x��  
�          @Ǯ���\@33?�ffA���Cٚ���\?���@\)A�G�C(�\                                    Bx�x�2  
�          @�\)��z�@�?���A�G�C����z�?��@0  A��C&�                                     Bx�x��  
(          @�p���G�@S�
?��HA4��C� ��G�@\)@�RA�z�C�                                    Bx�x�~  
�          @�G���@J=q?�
=AS�C����@G�@'�A���CW
                                    Bx�x�$  T          @�(����R?�=q?���A��\C'h����R>�G�@p�A�=qC/�q                                    Bx�x��  �          @����
=?��H?��RA���C(}q��
=>��@z�A�C1u�                                    Bx�yp  �          @�(����R?�=q@
=qA��\C"B����R?@  @.�RA���C,�                                    Bx�y  T          @����\)?���@
=A�33C$Y���\)>�@4z�A��HC/aH                                    Bx�y"�  �          @������?.{@�HA�C-k����;���@\)A�z�C7�                                    Bx�y1b  
�          @�z���ff>�=q@p�A���C1\)��ff�.{@�A�  C:�=                                    Bx�y@  
(          @�  ����>�ff@8Q�A���C/�=�����L��@333A̸RC;��                                    Bx�yN�  	`          @�����=��
@"�\A��RC3:����Ϳ�  @A�
=C=�H                                    Bx�y]T  
(          @�(����>.{@'�A�  C2T{��녿s33@��A�z�C=J=                                    Bx�yk�  
Z          @�z����H��Q�@$z�A��C4�����H��@�\A�Q�C?J=                                    Bx�yz�  
Z          @�����+�@
=A�
=C:}q�������?���A���CC0�                                    Bx�y�F  
(          @�����=q��G�@�A��
C@@ ��=q�
=?��A_
=CG�                                    Bx�y��  
(          @ʏ\��p��fff@�\A���C<�f��p���
=?�(�AW
=CC�{                                    Bx�y��  4          @�Q����Ϳc�
?���A�\)C<�=���Ϳ�=q?�ffA?�
CC�                                    Bx�y�8  �          @�  ��녿�ff@�A���C>L���녿�?��AM�CET{                                    Bx�y��  4          @�33��\)��\)?ٙ�A��C?���\)��p�?���A!CD�{                                    Bx�y҄  �          @�������333@z�A��
CQ�����Z=q?0��@��
CW
=                                    Bx�y�*  �          @\��Q��B�\@p�A��CT�)��Q��l��?8Q�@���CY��                                    Bx�y��            @�=q����{@z�A�33CNc�����N{?��A!��CT��                                    Bx�y�v  4          @��R����.�R@�A���CQY�����W�?E�@���CV�=                                    Bx�z  �          @�G�����7�?�=qA��RCRJ=����W�>�G�@�  CV}q                                    Bx�z�  �          @������n�R?�A��RC[O\�������=u?��C^L�                                    Bx�z*h            @�33����Y��?���Ar�\CW:�����p      =#�
CY��                                    Bx�z9  �          @�=q����fff?��HA�p�CZW
����\)<��
>aG�C]5�                                    Bx�zG�  B          @���~{����@
�HA��
Ca}q�~{��=q>�  @��Ce�                                    Bx�zVZ  	�          @�33������G�@��A�Q�Ca\�������>aG�@Cd��                                    Bx�ze   �          @��R�e����?���A�Ce���e��{�L�;�Ch��                                    Bx�zs�  �          @��R�G���G�?�Ax��C��q�G���p��\(����C��R                                    Bx�z�L  �          @�ff�@���\)@?\)B
=CX���@���S33?�  A�p�Cc�)                                    Bx�z��  �          @�(��5��!�@\)B(�C]Ǯ�5��Vff?�
=AqG�Ce�=                                    Bx�z��  �          @�����(��}p�@*�HA��C@�R��(���?��RA�p�CL�
                                    Bx�z�>  �          @��H��=q��\@1G�A��HCM8R��=q�AG�?��A�ffCV�3                                    Bx�z��  �          @���P���Vff@-p�A�RCaǮ�P����{?��
A/\)Ch�                                    Bx�zˊ  �          @��\��Q쿗
=?�\)A��CB�H��Q��G�?xQ�A5�CIG�                                    Bx�z�0  �          @����(��!G�?��\A\z�C;  ��(���{?aG�A(�C@33                                    Bx�z��  �          @�����=q��z�?�@ƸRC@�
��=q���<��
>aG�CB=q                                    Bx�z�|  �          @�Q����
����?�Q�AL��CI�����
�G�>�  @)��CL�q                                    Bx�{"  
�          @�ff�n{�_\)?�{A�z�C_!H�n{�|��>L��@�Cb�                                    Bx�{�  
�          @�{�U�k�?���A�CcǮ�U���>L��@
=qCgB�                                    Bx�{#n  �          @�Q��r�\�!�?�33A���CU���r�\�?\)>�
=@�\)CZ@                                     Bx�{2  �          @�z��u���?޸RA��CQٚ�u�.�R?!G�@�CWu�                                    Bx�{@�  �          @�����\)�
�H?˅A���CO33��\)�(��?�\@��\CT                                      Bx�{O`  �          @�=q��p���?�z�AU�CM�f��p���H>��?ٙ�CP��                                    Bx�{^  �          @�z���33��
=���
��  CEJ=��33��G��0���
=CCO\                                    Bx�{l�  �          @��\���
�����#�
��CC�����
��  �B�\�33CA�                                    Bx�{{R  �          @�����{����W
=�{C>����{�Y���#�
�ӅC<޸                                    Bx�{��  �          @�(���=q�.{�����HC:���=q��녿8Q����HC85�                                    Bx�{��  �          @�
=��{��
=�L����C8E��{������Q��j=qC7                                      Bx�{�D  �          @����{�8Q�=�Q�?p��C;+���{�0�׾aG��p�C:��                                    Bx�{��  T          @�(���33��>#�
?�C8޸��33�   �u�\)C9�                                    Bx�{Đ  
(          @����p��Y���u��RC<���p��+��z���G�C;�                                    Bx�{�6  �          @��H��33��=#�
>�
=CL���33���p���ffCJ0�                                    Bx�{��  �          @��\����(�>�  @#33CN
=�����
�Q��\)CLٚ                                    Bx�{��  T          @����p��QG�?\)@�=qCW  ��p��L(��c�
��CV^�                                    Bx�{�(  �          @������H�A�?B�\@�=qCT���H�C�
�(�����CTO\                                    Bx�|�  �          @�G������S33?��A_�CX)�����e������p�CZY�                                    Bx�|t  T          @����{�*=q?��HA�p�CPQ���{�H��>��@\)CTh�                                    Bx�|+  �          @��
���H�1G�@ffA�
=CS+����H�`��?p��A33CYn                                    Bx�|9�  �          @�z���=q��@7
=A�\)CO���=q�U�?˅A{�CX�                                    Bx�|Hf  �          @�(������?\)@�\A�=qCV#������k�?J=q@�(�C[�)                                    Bx�|W  T          @�Q���z��Y��?8Q�@��CV�=��z��XQ�L������CV�f                                    Bx�|e�  �          @�{��p����=�?�CK����p��p��h���CJ5�                                    Bx�|tX  
          @���{�A�>�p�@i��CSxR��{�8�ÿu�  CRW
                                    Bx�|��  
}          @�����\)�8��?=p�@��HCP���\)�;��\)���
CQE                                    Bx�|��  �          @�����p��H��?�  AffCS@ ��p��P�׾�G���=qCTE                                    Bx�|�J  �          @�{��z��S�
?p��A\)CTǮ��z��X�ÿ�����CUk�                                    Bx�|��  �          @�����P��?xQ�A��CT@ ����W
=��\��Q�CU\                                    Bx�|��            @�33����\)>��H@��CL������Ϳ�R���CK�
                                    Bx�|�<  �          @�����
=�p�?��@�ffCL:���
=��R��\���
CLff                                    Bx�|��  �          @����z���?(�@��CH����z�����Q��X��CI
                                    Bx�|�  �          @�  ��{��=q?�\@��CE����{��\)���
�B�\CF8R                                    Bx�|�.  T          @����p�����>���@<(�CBO\��p���Q쾳33�Y��CB5�                                    Bx�}�  �          @�����ff�c�
������C<޸��ff�@  ���H���C;�                                    Bx�}z  T          @�\)��{���������C:\��{�   ��p��j�HC8�R                                    Bx�}$   �          @��R����ff���R�HQ�C8�=����=q�������C6                                    Bx�}2�  "          @�p����;#�
��33�aG�C5�
����<#�
�\�w
=C3�                                    Bx�}Al  "          @�����>�ff�   ��G�C/c����?�R����%C-�                                    Bx�}P  
�          @�����\>��
��ff��z�C0�R���\>���=q�333C/�                                    Bx�}^�  	`          @�33��=q=L�;��H���C3u���=q>����
=����C1aH                                    Bx�}m^  �          @��
���H>���
=���C2� ���H>\�������C0{                                    Bx�}|  
�          @�p����\��\�E����HC95����\���h�����C5W
                                    Bx�}��  �          @�\)��p��Ǯ�+���\)C7����p��u�E����C4��                                    Bx�}�P  �          @��\���þ�Q�!G���C7����ýL�Ϳ8Q����HC4��                                    Bx�}��  �          @�p����\�n{��\����C=\���\��R�Y�����C:)                                    Bx�}��  �          @�33��ff��(���Q��e�C@���ff�s33�Y���{C=n                                    Bx�}�B  T          @������Ϳ�p���Q��dz�C@=q���Ϳs33�Y����HC=�{                                    Bx�}��  �          @��H��  �fff��Q��aG�C<�
��  �(�ÿ5��ffC:��                                    Bx�}�  
�          @�(���녿녿+����HC9�)��녾k��Y�����C6@                                     Bx�}�4  
�          @�{��33���c�
�	�C9���33��Q쿃�
�
=C4�)                                    Bx�}��  �          @�(����þ�G���  ���C8ff����<������+�
C3�q                                    Bx�~�  
�          @��\��
=��zῌ���.=qC6�f��
=>W
=��\)�1G�C1��                                    Bx�~&  �          @��H��{�aG���G��HQ�C633��{>�{���R�C�
C0�\                                    Bx�~+�  
�          @�=q��{��Q쿙���>�\C4޸��{>�(���\)�2{C/��                                    Bx�~:r  %          @�G�������Ϳ��R�FffC5����>�G����:=qC/��                                    Bx�~I  �          @�ff��  ���\�v�RC4J=��  ?�R�����`��C-��                                    Bx�~W�  %          @����R��G�����{\)C5+����R?�Ϳ�Q��j�RC.5�                                    Bx�~fd            @�
=�����
���
���C4@ ��?=p���\)��{C,=q                                    Bx�~u
  �          @�33��Q�>��H����\)C.�H��Q�?�p������pz�C&�{                                    Bx�~��  �          @�����(�?녿��R��\)C-���(�?�\)��ff�z=qC%                                    Bx�~�V  �          @�\)��ff>����H��\)C2�3��ff?Y�����R�g�
C+�\                                    Bx�~��  �          @��R��{>8Q��(�����C2=q��{?c�
��p��g
=C+
                                    Bx�~��  T          @��H�����#�
�ٙ�����C4�����?8Q����l(�C,�                                    Bx�~�H  �          @�33��ff�G����
��C;k���ff�����G��?33C6�=                                    Bx�~��  �          @�ff��
=�W
=��\)�}G�C6)��
=?   �Ǯ�s
=C/                                      Bx�~۔  �          @�{��p���Q��  �c\)C7n��p�>��
��G��d��C0�                                    Bx�~�:  �          @ƸR���\>\�z����C0:����\?�p����H��C()                                    Bx�~��  �          @�\)��\)?(��ff��p�C-�f��\)?Ǯ��\)���C$��                                    Bx��  �          @�����W
=�����{C6(����?Tz��G�����C+�f                                    Bx�,  �          @ƸR��Q�?s33�%����C*8R��Q�?��H��Q����HC c�                                    Bx�$�  �          @�����H@z��Q���C�����H@G
=���+
=C��                                    Bx�3x  T          @�����(�?s33�H����Q�C*xR��(�@(��(���C�                                     Bx�B  �          @ҏ\����?L���@�����
C,)����@   �Q����C �f                                    Bx�P�  c          @љ�����=�Q��E���RC3�����?���0  ���HC&u�                                    Bx�_j  	�          @��
���>aG��:�H��=qC1�����?�Q��#33���HC%                                    Bx�n  �          @˅��\)>�ff�.{����C/� ��\)?Ǯ�G����C$�R                                    Bx�|�  �          @�33���?#�
�G
=��RC-\)���?�33�"�\����C �                                    Bx��\  T          @ȣ���Q�?�\�<������C.�3��Q�?�(��p����\C"��                                    Bx��  �          @�33����?5�AG���\)C,�����?�
=����Q�C ��                                    Bx���  �          @��H����>�{�G
=���HC0xR����?У��*�H��G�C#�\                                    Bx��N  �          @�33����>��E����C/�����?޸R�%��
=C"�                                     Bx���  "          @Ǯ��
=>�\)�>{��p�C1���
=?\�$z��\C$�                                    Bx�Ԛ  "          @�{���>8Q��5��
=C2#����?�\)�   ���RC&                                    Bx��@  �          @�\)����>��
�6ff��=qC0�����?�  �����G�C$�\                                    Bx���  �          @���
=>����7���33C0ٚ��
=?��R�{��z�C$�                                    Bx�� �  "          @�G���{>k��$z����
C1����{?�ff��R����C&�{                                    Bx��2  �          @�\)���H>��+��ՅC2�����H?�  �Q���p�C&Ǯ                                    Bx���  �          @�
=��=L���<�����C3h���?�ff�)����
=C%��                                    Bx��,~  T          @�����aG��(����C6=q���?:�H����ffC,ff                                    Bx��;$  T          @�������  �33��
=C6����?#�
���H��C-5�                                    Bx��I�  "          @����
=����
�H���RC5�
��
=?J=q����Q�C+n                                    Bx��Xp  �          @�Q�������G���R��Q�C58R����?W
=�z���z�C+                                      Bx��g  �          @����
=��׿�����\)C8����
=>�Q�����p�C0B�                                    Bx��u�  �          @�Q���p���G���\)��  C8����p�>Ǯ�����
=C/��                                    Bx���b  �          @�����
=��  ��{���
C6����
=?\)�����  C.&f                                    Bx���  
�          @�Q����R=��
��=q����C3����R?Y���У���p�C+&f                                    Bx����  T          @��R��  �#�
��(��o
=C4u���  ?zῬ���[\)C-�q                                    Bx���T  T          @������\�#�
��
=�eG�C5�����\>���{�Z=qC/J=                                    Bx����  �          @�=q���׾���
=��(�C8Ǯ���׾���z���Q�C6�
                                    Bx��͠  �          @�G���ff�O\)>aG�@p�C<)��ff�Tz�\)���C<G�                                    Bx���F  T          @�33����ff?Tz�A33C>u�����ff>��R@E�C@�f                                    Bx����  �          @��
�����z�?���A=p�CD�
����G�>Ǯ@u�CH#�                                    Bx����  T          @�p����׿xQ�>�ff@�{C=�����׿���<#�
=L��C>z�                                    Bx��8  �          @��H��Q�G�>B�\?�z�C;���Q�J=q�����p�C;Ǯ                                    Bx���  �          @�p���녿0��?��@���C:�\��녿\(�>W
=@33C<p�                                    Bx��%�  T          @����zᾊ=q��\)�-p�C6����z����p��c�
C5B�                                    Bx��4*  T          @�z����
��\)�����=qC4�����
>#�
�����
C2c�                                    Bx��B�  �          @�33��ff�Y��>�(�@���C<}q��ff�s33=#�
>�Q�C=��                                    Bx��Qv  �          @��H������?fffA=qC>������\)>�Q�@b�\CA��                                    Bx��`  T          @�����Ϳc�
?
=q@�(�C<����Ϳ��
=�?�z�C>Y�                                    Bx��n�  �          @��H��z῜(�?fffAp�C@:���z῾�R>���@>{CB                                    Bx��}h  T          @��H�����?O\)@�
=C?�3�����33>u@��CA�                                    Bx���  �          @��
���H�\?\(�A�CC(����H�޸R>��?�(�CEB�                                    Bx����  T          @�(����ÿ���?��RAB�\CCٚ���ÿ�Q�>��@�=qCG^�                                    Bx���Z  T          @�=q���׿�ff?p��AQ�CC�����׿�ff>L��@   CF
=                                    