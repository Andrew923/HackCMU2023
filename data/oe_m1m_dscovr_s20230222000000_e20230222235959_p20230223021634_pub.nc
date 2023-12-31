CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230222000000_e20230222235959_p20230223021634_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-23T02:16:34.192Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-22T00:00:00.000Z   time_coverage_end         2023-02-22T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxgf��  
Z          A����@�(��w���G�C����@�<��
=�Q�B���                                    Bxgf�&  �          Az����@��[����HC �q���@��R?��@Q�B��\                                    Bxgf��  
�          A
=��(�@�
=�A����B����(�@�\)?�z�@��
B���                                    Bxgf�r  
�          A
=���@��L�����\B������@�
=?u@�{B�L�                                    Bxgf�  
�          A\)��Q�@�  ���0  B�8R��Q�@�p�@aG�A���C                                    Bxgg �  @          A Q����HA ��    <��
B������H@�{@�=qA�(�C �H                                    Bxggd  �          A*�R��{A
�H@(�A>�RB�p���{@�z�@���B�C�                                    Bxgg
  
�          A1����{A
=@��A2{B�aH��{@�z�@˅B�C                                      Bxgg,�  �          A4Q�����A{?u@�z�B�z�����@�@��A���C �                                    Bxgg;V  �          A3
=��p�A?(��@Y��B����p�@��R@�p�A�{B���                                    BxggI�  �          A5����
=AG�@
�HA0z�B����
=@���@ڏ\B33B�k�                                    BxggX�  "          A733���A  ?�Q�Az�B�8R���@�{@�33B�HB�z�                                    BxgggH  "          A-p�����A�H��\)��\B��
����Aff@Tz�A�=qB�\                                    Bxggu�  "          A-���
=A���33�/�
B����
=A�\@!�AY�B���                                    Bxgg��  "          A4  ��  A	���ff��ffCE��  A ��@L(�A�G�C�                                    Bxgg�:  "          A(z�����@�  �w�����C������@�G�=L��>���C��                                    Bxgg��  �          A!���
@���Å�z�C�����
@����,(��w\)C)                                    Bxgg��  T          A"=q���A{��z��-��B陚���A
{@4z�A��B��                                    Bxgg�,  �          A"=q���A���p��`z�B�W
���AG�@33AQB��                                    Bxgg��  
(          A"=q��ffA{�0  �|��B�33��ffA
{?�A'\)B�q                                    Bxgg�x  "          A!G���=q@�ff�r�\����B�p���=qAz�?8Q�@��B�                                    Bxgg�  �          A#
=��(�A���n{���
B�#���(�A�?\(�@��\Bힸ                                    Bxgg��  �          A-����p�A�����<z�B� ��p�A��@7
=AuB�\)                                    Bxghj  
�          A'����A�R�!��_
=B��=���A��@�\A3�B��R                                    Bxgh  "          A$�����@�z��N{��{C	&f���@�?��@C�
C33                                    Bxgh%�  �          A%������Az��%��h  B������A33?�
=A,��B��{                                    Bxgh4\  
�          A/
=��\)A���'
=�^�\B�p���\)Ap�@��AL��B�(�                                    BxghC  
�          A0  ����A����?�B�\����A��@2�\Ak33B�                                    BxghQ�  
�          A.{��(�A(��
=�IG�B�����(�A33@&ffA]B�Q�                                    Bxgh`N  r          A0  ���A����.�\B�33���A�@E�A�{B�u�                                    Bxghn�  T          A0�����A z���\�*=qB�{���A33@P��A�G�B�p�                                    Bxgh}�  "          A0�����A��33�*�HB������A�@J�HA��B�W
                                    Bxgh�@  �          A1p���=qAG����B{B��
��=qA�H@:=qAs
=B�=                                    Bxgh��  
�          A/33��(�A�\�������
B�3��(�A��@l��A�ffB�k�                                    Bxgh��  
�          A0  ��{A�׿\���RB�#���{AQ�@]p�A���B��H                                    Bxgh�2  
�          A0�����A���*�H�_�B�ff���A=q@AD(�B��                                    Bxgh��  T          A0������A  �5�n�\B����A�H@
�HA5�B��                                    Bxgh�~  "          A-�����Ap��B�\����B�q����A  @x��A��
B���                                    Bxgh�$  �          A,(���{A33�W
=��\)B�.��{AG�@��A�
=B��                                    Bxgh��  
�          A)G��ʏ\A
=�u����B��3�ʏ\@��H@�p�A�{C�q                                    Bxgip  "          A(Q���Q�A=u>�{B�aH��Q�@�
=@�
=A�C�                                    Bxgi  	�          A)G��w�A33@.{An�RB��=�w�@ڏ\@�=qB(  B�\                                    Bxgi�  "          A'\)���HA�@I��A�Q�B�\���H@��@�\B*\)C �3                                    Bxgi-b  �          A'����Az�@L��A���B�����@���@���B,B��=                                    Bxgi<  
�          A'33�j=q@��
@��BQ�B��
�j=q@Tz�A(�Bu
=C	Ǯ                                    BxgiJ�  
�          A$Q��>{@�33@��HB"(�B�k��>{@8Q�A  B�C��                                    BxgiYT  �          A(z����
@��H@�\)B=�CQ����
���
ABfz�C4��                                    Bxgig�  "          A+\)���@z=q@�
=B;�RC!H��녾�G�A
=qB[�\C8(�                                    Bxgiv�  T          A*�H��z�@���@׮B�C 5���z�@z�Ap�BhC��                                    Bxgi�F  �          A'����R@��H@�
=B��B�.���R@.{A(�Bo��C��                                    Bxgi��  "          A$z��e@�\@ǮB�\B�33�e@P��A��Bw=qC	�R                                    Bxgi��  "          A$(��b�\Az�@��\A���B�\�b�\@�Q�A   BN\)B��H                                    Bxgi�8  �          A&{�^{A
=@I��A�(�B�k��^{@�p�@�G�B3�B��)                                    Bxgi��  
�          A.=q��
=A=q@�A4��B�q��
=@�=q@��
B�RB���                                    Bxgi΄  
�          A,���~�RAG�?�z�@�\B�\�~�R@��H@�33B  B���                                    Bxgi�*  "          A.=q����A Q�?��@�  B�����A=q@�ffB�HB虚                                    Bxgi��  
(          A%���@��@
=AV�RB������@���@�Q�B  C!H                                    Bxgi�v  T          A$������@�@vffA��RB������@�33@ᙚB0G�C�)                                    Bxgj	  �          A)G����A�@0  Aqp�B�3���@�
=@ָRB\)C��                                    Bxgj�  
�          A(�����A�@(�AW\)B�R���@��
@�B(�C�{                                    Bxgj&h  
�          A&�H���Az�@#�
Ac\)B��q���@���@�{B��CL�                                    Bxgj5  �          A*=q���
A�@S33A�
=B� ���
@�p�@陚B-B���                                    BxgjC�  T          A*�H��
=A��@{AAG�B�3��
=@�=q@���B�C��                                    BxgjRZ  �          A+33��p�A�
@3�
At��B����p�@�
=@�Q�BG�C�                                    Bxgja   T          A+��\A�@e�A�{B�
=�\@�  @��HB%�C+�                                    Bxgjo�  
(          A*�H���A33@7�AyG�B� ���@��@���B��C��                                    Bxgj~L  
�          A+
=��p�A
�R@ ��A-�B��3��p�@��@�Q�B�C
                                    Bxgj��  
�          A*ff����Aff?�
=A{B��H����@�p�@��HB�C �R                                    Bxgj��  "          A*�H��ffAz�?�z�A33B�=q��ff@�G�@�(�BB�p�                                    Bxgj�>  
�          A)�����HA\)@:�HA�z�B�����H@�@�ffB��CO\                                    Bxgj��  T          A(����\)A@EA�G�B�z���\)@���@ٙ�B"  C��                                    BxgjǊ  �          A+
=��33A��?��RA ��B��f��33@�ff@���B  B�B�                                    Bxgj�0  
�          A&�R����A
=?���A-G�B�z�����@��H@\Bz�B�L�                                    Bxgj��  
�          A$����=qA�\?h��@��\B�L���=q@�=q@��A�RB�k�                                    Bxgj�|  
�          A$  ���RA�R?��@��
B�=q���R@��@��\A��C�R                                    Bxgk"  T          A$(�����A(�>�33?���B�(�����@�
=@��HA�33B��                                    Bxgk�  "          A*{����A=�Q�>�B�ff����A�@���A�  B�p�                                    Bxgkn  
Z          A(z�����A�R���+�B������A ��@�(�Aģ�B�.                                    Bxgk.  T          A&{��z�A{�+��mp�B�W
��z�A��@w�A�G�B�
=                                    Bxgk<�  �          A$(���=qA�������
B� ��=qA��@Q�A�p�B�8R                                    BxgkK`  
�          A$  ��AG��\�z�B�\)��A33@>�RA��B��=                                    BxgkZ  
�          A�����
A녿����z�B�L����
@�33@%A}��B��                                    Bxgkh�  
�          AQ����@�������'�
B�
=���@�@�Ar�HB�8R                                    BxgkwR  T          A33���@�
=�����$��B����@��@
=Atz�B�Ǯ                                    Bxgk��  
�          Az���33@�=q@H��A��B�� ��33@��\@�B$=qCu�                                    Bxgk��  S          A����@�  @{�A�{B������@�z�@�\B7��C	s3                                    Bxgk�D  T          A{���
@�(�@��RA�=qB�  ���
@�@���B9�\C�                                    Bxgk��  �          A����\@陚@g�A��B�Ǯ���\@�(�@ָRB+�CT{                                    Bxgk��  
Z          A���(�@��>�z�?���C�R��(�@��@c�
A���C}q                                    Bxgk�6  T          A���H@�(��ٙ��'�C}q���H@��H?���A6�HC��                                    Bxgk��  
�          A�R��p�@�G����ÿ��RC@ ��p�@�=q@G�A��
CW
                                    Bxgk�  "          A��G�@�\�aG���{C O\��G�@У�@VffA�=qC�)                                    Bxgk�(  T          A�H����@�33�u��33C �q����@θR@`  A���CaH                                    Bxgl	�  T          A���ȣ�@��
�Y�����CaH�ȣ�@�@$z�A�
CJ=                                    Bxglt  
�          A
=��\)@��H��G���B�����\)@�z�@1�A�G�B�                                    Bxgl'  �          A  ��@�Q�p�����RB�33��@�G�@333A��C �                                    Bxgl5�  
�          Az���ff@�ff���
��ffB��=��ff@�@&ffA}�B�                                      BxglDf  "          A(���ff@��
��G���\B��f��ff@�  @,(�A�p�B�p�                                    BxglS  
Z          A�
��Q�A (����I��B�\��Q�@���@X��A��HB�G�                                    Bxgla�  
Z          A{��{@��>�@1G�C Q���{@��
@���A�Q�C�                                    BxglpX  �          A�����@��=u>ǮB������@���@w
=A�
=B�33                                    Bxgl~�  "          A���H@��?fff@��
B����H@�Q�@��HA�RC\)                                    Bxgl��  
�          Aff�˅@��@W
=A��\C:��˅@z�H@�\)B��CW
                                    Bxgl�J  �          A
=���H@��
@7�A�
=C����H@�
=@�=qB�HC^�                                    Bxgl��  
�          Aff��33@�  @*�HA��CY���33@U�@��A�
=CW
                                    Bxgl��  "          A�R�z�@�(�?�Q�A"{C0��z�@3�
@]p�A���C!                                    Bxgl�<  
�          A���@��
?@  @�ffC���@z�H@?\)A��C�\                                    Bxgl��  "          A���
=@���@$z�At��C�{�
=@=q@��A�Q�C$�                                    Bxgl�  �          A�����@n�R@C33A���Ch����?���@��RA�ffC'��                                    Bxgl�.  
�          AQ��
�H@-p�@Z=qA��\C"�f�
�H?@  @���A��
C/�                                    Bxgm�  �          A!G��
=@:=q@   Ah(�C"h��
=?�\)@e�A���C+�                                    Bxgmz  "          A"ff���Az�?!G�@a�B��)���@�p�@��A�33B���                                    Bxgm    
�          A$z�����A�?O\)@���B�����@��
@��A��HB��                                    Bxgm.�  �          A#\)��\)@�p�@�A?\)CB���\)@��H@��
A�33C޸                                    Bxgm=l  �          A z��@�Q�@n{A�
=C\)�?�G�@�A�  C(�                                    BxgmL  �          A#
=�  @�G�@xQ�A�\)C�
�  ?��H@��\A���C(��                                    BxgmZ�  
�          A%�����@�\)@�
AN=qC\)���@|��@�  Aڣ�C��                                    Bxgmi^  "          A)p����R@�?fff@�(�C
�����R@�@��\A��C��                                    Bxgmx  �          A(������@�\?\(�@��C	������@��\@��A�\)C�                                    Bxgm��  T          A)G����H@���?��@���C
!H���H@�ff@��A��C��                                    Bxgm�P  
�          A*=q��
=@�  ?n{@���C���
=@�ff@�  A�z�CY�                                    Bxgm��  �          A)��G�@�{?8Q�@x��C^���G�@�
=@���A�  Cz�                                    Bxgm��  
�          A)����\@��
?�@6ffC�f��\@�
=@��A�Q�C
�{                                    Bxgm�B  
�          A)��  @��>�{?���C(���  @��@|(�A��\C	n                                    Bxgm��  T          A)��@��>��@�RC\��@�z�@�Q�A��C	z�                                    Bxgmގ  �          A)p����
@�=q>�Q�?��HCT{���
@�\)@�  A��C��                                    Bxgm�4  
�          A*{�陚@�ff?   @+�Cs3�陚@�=q@��A��C
                                    Bxgm��  
�          A+
=��(�@��R?��@N�RC����(�@���@���A��HC
�                                     Bxgn
�  �          A+���
=@���?E�@�33CL���
=@���@���A���Ch�                                    Bxgn&  T          A*=q��R@���?Tz�@�  C��R@�Q�@�Q�A��C                                    Bxgn'�  �          A)����@�=q?B�\@��CQ�����@��H@�
=A�\)Ch�                                    Bxgn6r            A)G����@�33?Tz�@�  C�f���@ʏ\@�G�A�G�C!H                                    BxgnE  �          A)��陚@�33?O\)@��C�\�陚@��H@���A��RC
�q                                    BxgnS�  T          A)p���33@�=q?c�
@���C(���33@���@�=qA�z�C}q                                    Bxgnbd  �          A)���@�\)?z�@I��C�f��@��@��
A��C	��                                    Bxgnq
  T          A((����H@�  >��@\)Cs3���H@�p�@~�RA�C                                    Bxgn�  T          A)p��޸R@�
=<�>.{C��޸R@�G�@o\)A��C�f                                    Bxgn�V  �          A+33���
A
=���8Q�C���
@��@l(�A�p�C=q                                    Bxgn��  "          A+33��R@�z�>�(�@z�Cn��R@�G�@���A��C�                                     Bxgn��  �          A+33��ff@�(�?z�@G�Cp���ff@ָR@�p�A�C	
=                                    Bxgn�H  �          A+
=���A�>��
?�p�C�)���@���@�33A��\C��                                    Bxgn��  T          A+\)��
=AG�>�z�?���C �f��
=@�  @��
A���Cٚ                                    Bxgnה  
=          A*�H����AQ�?
=@I��B��)����@���@��RAř�CT{                                    Bxgn�:  �          A+33�ǮA
ff?Q�@�z�B����Ǯ@�G�@�
=Aљ�C�{                                    Bxgn��  "          A+�����A
{?aG�@�{B������@�  @�Q�A�33C�f                                    Bxgo�  
�          A+\)�˅A��?p��@��B�G��˅@��@���A�Q�C�f                                    Bxgo,  "          A*�H�ə�A��?�=q@���B�Ǯ�ə�@��H@���A�ffC��                                    Bxgo �  
�          A+�
��=qA	�?�=q@�(�B��)��=q@߮@�(�A�=qC!H                                    Bxgo/x  
�          A+�����AQ�?��H@�\)B��
����@�Q�@��A��Cff                                    Bxgo>  "          A+\)����A
=?���A	G�B�\)����@أ�@���A�
=Ck�                                    BxgoL�  T          A,Q���  Az�?�{A	�B��3��  @��@��RA�  CY�                                    Bxgo[j  7          A,Q���p�A�?�  @�B�����p�@�@���A�z�B�33                                    Bxgoj  �          A,  ��A{?˅A�
B� ��@�z�@��HA��HB���                                    Bxgox�  "          A,������A�R?��A��B�������@��H@�G�B�RB��R                                    Bxgo�\  7          A-������A�R?�p�A
=B�\����@��
@�\)A�33B���                                    Bxgo�  
�          A-����=qA��?ٙ�Az�B�\��=q@�  @�Q�B 33B��R                                    Bxgo��  T          A,����
=A  ?��A!��B��
��
=@��
@���B��B��\                                    Bxgo�N  T          A+����A��?���A33B������@��
@��A�ffC��                                    Bxgo��  "          A+����A
�H?˅A��B��)���@�Q�@��
A��\C�)                                    BxgoК  	�          A,����@���>���?���C�{��@�@p  A�ffC
��                                    Bxgo�@  �          A,���ff@�Q��(���\C0��ff@�
=@+�Af=qC�                                     Bxgo��  "          A,����\)@�G�>#�
?^�RC����\)@���@c�
A�(�CL�                                    Bxgo��  i          A-G���Q�@�G�<��
=�C�{��Q�@�
=@[�A�  C#�                                    Bxgp2  ?          A,����\@�{=�G�?��C����\@��@e�A���C
\                                    Bxgp�  "          A+���@�33�u���
C�\��@��@XQ�A�
=C	�3                                    Bxgp(~  
�          A*{�陚@��R<�>#�
Ck��陚@��
@`��A�=qC�q                                    Bxgp7$  T          A*�\��z�@��>\)?E�C�3��z�@���@e�A�G�C	xR                                    BxgpE�  
�          A*ff��(�@��?
=q@;�C���(�@�33@{�A�p�C
5�                                    BxgpTp  
�          A(����\)@�G��(���hQ�Cٚ��\)@�Q�@:=qA���C�H                                    Bxgpc  �          A(  ��G�@��?h��@�
=C�f��G�@��@��A�
=C��                                    Bxgpq�  �          A(z����HAff?Tz�@�  C ����H@�p�@�z�Ař�C�)                                    Bxgp�b  �          A)G���z�A
=?(�@Tz�C���z�@ᙚ@�
=A�=qCG�                                    Bxgp�  
�          A(����{Az�?�\)@��HB�Ǯ��{@���@�ffA��
C                                    Bxgp��  �          A(z����HA  ?У�A�\B�����H@���@���A�  C�H                                    Bxgp�T  �          A)G��ÅA�R?�\)A#\)B���Å@�ff@�{A��C\)                                    Bxgp��  �          A,z�����A
{?��A33B�{����@�G�@�
=A�C��                                    Bxgpɠ  �          A,����\)A��?��\@�{B�k���\)@�{@�
=A�C
=                                    Bxgp�F  �          A,Q���A
==���?�\C c���@�Q�@uA�
=C��                                    Bxgp��  "          A*ff���HA>�\)?�p�C =q���H@��
@}p�A���C��                                    Bxgp��  �          A)p��ָRA
=>���?�G�CW
�ָR@�{@z�HA��C�                                    Bxgq8  i          A(z��ҏ\Aff?xQ�@��C ��ҏ\@�(�@�\)A�C                                    Bxgq�  
E          A*�R��{AQ�?�@B�\C ����{@�p�@�p�A�(�C
=                                    Bxgq!�  
�          A+
=�ۅA�R��z����C�ۅ@�R@VffA�Q�C�)                                    Bxgq0*            A*�R���
A{�.{�hQ�C33���
@��H@>�RA��C!H                                    Bxgq>�  
w          A)p��޸R@�
=>aG�?�
=C#��޸R@��@mp�A�{C�)                                    BxgqMv  T          A)���=q@�=q=�?#�
C���=q@�
=@c33A�\)Cc�                                    Bxgq\  �          A(������@�Q�\�z�C�����@�z�@Dz�A�33C�                                    Bxgqj�  T          A(���ᙚ@��\��(��
=C  �ᙚ@�\)@C33A�ffCJ=                                    Bxgqyh  
�          A(����{@�
=��(��C�3��{@�(�@@  A�  C:�                                    Bxgq�  �          A(�����
@񙚾���#33CQ����
@߮@8Q�A}p�C��                                    Bxgq��  �          A(����\@�녿��6ffC!H��\@���@5�Ayp�C8R                                    Bxgq�Z  
�          A(  ����@�{��p���\C�
����@�33@:=qA��HC	.                                    Bxgq�   
�          A&�H��(�@�33��G���C!H��(�@��@3�
Az{C	T{                                    Bxgq¦  �          A'33���
@�z�z��J=qC�f���
@��@,��Ao�Cٚ                                    Bxgq�L  T          A&�H����@�p��W
=���Cu�����@�G�@{A[�C�3                                    Bxgq��  
�          A&�\��p�@��xQ����\C�q��p�@�p�@Q�AS�C                                      Bxgq�  �          A&�H��G�@�(���{�ÅC����G�@�z�@��AC
=C��                                    Bxgq�>  �          A&�\�ڏ\@��R��z���=qC���ڏ\@��@z�A7\)C�                                    Bxgr�  T          A&=q���
@��\?
=@VffC����
@ٙ�@z�HA��\C)                                    Bxgr�  �          A'33��\)@�ff?W
=@��\C:���\)@��@�{A��C�f                                    Bxgr)0  
�          A'
=���@��>Ǯ@�C����@�(�@n{A���C&f                                    Bxgr7�  �          A&�\�޸R@��>u?�ffC���޸R@�(�@c33A�p�CO\                                    BxgrF|  
(          A&ff��33@�=q>���?�C5���33@�@h��A��
C��                                    BxgrU"  "          A&ff��33A Q�?�R@Z=qCxR��33@�
=@�  A��Cn                                    Bxgrc�  �          A&{�љ�A Q�?+�@l(�CE�љ�@�ff@�G�A��CO\                                    Bxgrrn  �          A%����Q�A Q�?0��@s�
C���Q�@�ff@��A���C)                                    Bxgr�  
�          A%p���@�(�?��@C�
CE��@�z�@w�A��
C�                                    Bxgr��  �          A&�\��\)@�\)?�\@3�
C���\)@�G�@j=qA��\C	�
                                    Bxgr�`  
=          A%p�����@�?+�@mp�C������@��@xQ�A�ffC�q                                    Bxgr�  
�          A$���׮@�Q�?W
=@�p�C��׮@��@��A��C^�                                    Bxgr��  �          A$z��ۅ@��H?Tz�@��
C��ۅ@У�@~�RA�ffCz�                                    Bxgr�R  
�          A&�\����@�33?W
=@�z�C�{����@�  @�33A�p�C&f                                    Bxgr��  T          A&=q���A z�?��@AG�C=q���@�G�@y��A���C�R                                    Bxgr�  �          A'
=��  A�\>�33?�z�C �\��  @�@qG�A��C�                                    Bxgr�D  �          A'33��33Ap�>�@   C8R��33@�z�@uA�=qC�                                    Bxgs�  T          A'
=�θRA
=?   @.�RC @ �θR@�R@z=qA�{C�{                                    Bxgs�  "          A&�H���
A�>��H@*�HB�k����
@�Q�@z=qA��\C=q                                    Bxgs"6  i          A&�\��A�\?\)@FffC =q��@��@|��A�ffC�                                    Bxgs0�            A&�H�љ�Ap�?�R@Z�HC ���љ�@�\@~{A��HC�                                    Bxgs?�  T          A&ff��Q�Ap�?(�@W�C ����Q�@��H@|��A���C�\                                    BxgsN(  T          A%G���(�A��?\)@FffC :���(�@��
@y��A��C޸                                    Bxgs\�  T          A$�����A��?
=@Q�B��H���@�@{�A��C�)                                    Bxgskt  "          A$����=qAp�?(��@j=qC ��=q@�=q@~�RA�{C��                                    Bxgsz  
�          A$z���Q�@�?333@{�CaH��Q�@��@|��A���CG�                                    Bxgs��  T          A$(���@��R?=p�@�z�C ����@�p�@\)A��C�f                                    Bxgs�f  �          A$����ffA (�?@  @��C �
��ff@�
=@�Q�A�G�C�                                    Bxgs�  "          A#���  @��?E�@���C�)��  @ڏ\@~{A�ffC��                                    Bxgs��  
�          A#
=��z�@��?.{@s�
C �3��z�@��@y��A��C                                    Bxgs�X  "          A%���љ�@�
=?J=q@���Ck��љ�@�p�@���A�33Ch�                                    Bxgs��  	�          A%G���=q@�{?=p�@�33C�)��=q@�p�@}p�A�(�C��                                    Bxgs�  T          A%p���{@��H?E�@�G�Cz���{@�=q@|(�A�\)Cs3                                    Bxgs�J  T          A%G���ff@��\?=p�@��\C����ff@ڏ\@y��A�\)CxR                                    Bxgs��  	�          A%G���G�@�
=?�G�@��CO\��G�@��
@�33A��C��                                    Bxgt�  
�          A$(���(�@���?Y��@�\)Cu���(�@׮@~{A�(�C�=                                    Bxgt<  "          A$���Ϯ@�\)?&ff@fffC
�Ϯ@�Q�@w�A�Q�C                                    Bxgt)�  
�          A$���љ�@�{>�Q�@ ��C���љ�@�33@eA���C��                                    Bxgt8�  "          A%���Ӆ@�ff>��H@.{C�q�Ӆ@ᙚ@l��A��C#�                                    BxgtG.  
�          A%p��ָR@�33?�@8Q�C}q�ָR@�ff@k�A���C��                                    BxgtU�  �          A$����p�@��\?�@L��Cff��p�@�p�@n{A�\)C�                                    Bxgtdz  �          A$(����
@��?.{@tz�CB����
@ۅ@s�
A�=qC��                                    Bxgts   T          A"=q��z�@��
?Tz�@�C���z�@��
@vffA��RC�                                    Bxgt��  �          A"=q��\)@�?5@���C�3��\)@Ӆ@n{A�{C�                                     Bxgt�l  
�          A"=q�ָR@��?Q�@�(�C���ָR@ҏ\@s�
A���C�
                                    Bxgt�  
�          A#33�ڏ\@���?\(�@�33C.�ڏ\@�G�@uA���C:�                                    Bxgt��  �          A"�R��Q�@��?Y��@�=qC�\��Q�@�=q@uA�33C�{                                    Bxgt�^  
Z          A#���p�@�ff?8Q�@���C����p�@�Q�@qG�A��C�f                                    Bxgt�  �          A"�R���H@��R?(��@l��C����H@�G�@mp�A�
=C!H                                    Bxgt٪  "          A#\)��z�@�ff?   @1�C ����z�@�\@i��A���C{                                    Bxgt�P  �          A!���=q@�p�?Y��@�=qC����=q@θR@p��A��\C�\                                    Bxgt��  �          A �����@��
?��@���C�{���@�33@tz�A��RC                                      Bxgu�  �          A!���=q@��?��
@陚C���=q@��R@�  A�33Cٚ                                    BxguB  �          A (���@Ӆ?˅A�\C
\��@�p�@�33A�\)C�)                                    Bxgu"�  
�          A\)���
@�Q�?��
AC
�����
@��@�  A��C�q                                    Bxgu1�  
�          A�H��z�@��?ٙ�A��C���z�@��R@��AîC��                                    Bxgu@4  
Z          A�
��  @Ӆ?�z�AC	�f��  @���@���A�G�CL�                                    BxguN�  "          A ������@��?�Q�@��C������@ʏ\@�G�A�Ck�                                    Bxgu]�  
�          A ���θR@�(�?}p�@�C=q�θR@Ӆ@|(�A��\CQ�                                    Bxgul&  �          A   ��ff@�?�z�@��C� ��ff@�\)@���A�G�C�H                                    Bxguz�  "          A
=��Q�@�?�p�@�(�Cs3��Q�@�Q�@�z�A���C��                                    Bxgu�r  T          A �����@�\)?�  A	�C	0����@�33@�G�A��Ch�                                    Bxgu�  T          A!G���  @�=q?�=q@�=qC����  @���@mp�A���C�3                                    Bxgu��  
�          A!G����R@�G�?˅Ap�C�����R@���@{�A��
C=q                                    Bxgu�d  �          A!G�����@�Q�?˅AG�C!H����@��
@���A��C�                                    Bxgu�
  �          A (���G�@�{?��RA	G�C	xR��G�@�=q@\)A�(�C��                                    BxguҰ  �          A���
=@�\)?���Ap�C	  ��
=@�(�@}p�A�G�C�                                    Bxgu�V  T          A!p���p�@޸R?��@��
C�)��p�@�z�@|��A�ffC��                                    Bxgu��  "          A (���@��?�33A (�C����@�=q@~�RA�G�C�3                                    Bxgu��  �          A!���{@��?�33@�
=C���{@��\@~�RA�  C��                                    BxgvH            A Q���z�@�(�?�33A ��C���z�@���@}p�A��\C�f                                    Bxgv�  �          A
=����@��?��
@���Cz�����@��
@w
=A���C)                                    Bxgv*�  
Z          A33��33@�p�?@  @���C�3��33@�=q@]p�A���C	L�                                    Bxgv9:  
�          A (���\)@�(�?�\A#33C����\)@�ff@�33A���C33                                    BxgvG�  T          A!����{@�33?޸RA�HC
k���{@�p�@���A�=qC�                                    BxgvV�  T          A ����p�@��?�\)@�=qC���p�@\@���A�z�C
�3                                    Bxgve,  �          A ����z�@�p�?��RA  C�
��z�@���@�(�A��C
�3                                    Bxgvs�  T          A!����z�@��?�G�@���Cz���z�@�\)@|��A�(�C	�H                                    Bxgv�x  T          A!G�����@�  ?�p�@�\)C������@�\)@z=qA�z�C	��                                    Bxgv�  �          A!�����
@�(�?z�@R�\C�����
@��H@VffA�33C.                                    Bxgv��  �          A!�����@�>�ff@#33C.����@�  @QG�A�
=C&f                                    Bxgv�j  T          A"�R��\)@�>�@%�C� ��\)@ۅ@Tz�A�z�Cs3                                    Bxgv�  "          A"ff��Q�@���?=p�@�ffC����Q�@�@c33A��
CT{                                    Bxgv˶  �          A"�R��z�@�ff?B�\@�G�C� ��z�@�33@a�A�z�C33                                    Bxgv�\  �          A"=q�ۅ@�{?\)@J�HC���ۅ@�@UA�C�=                                    Bxgv�  T          A"�R���@�p�?(��@l��C����@�(�@Z=qA�
=C33                                    Bxgv��  
�          A"{�أ�@�  ?L��@���C\�أ�@�z�@dz�A���C��                                    BxgwN  
�          A"�H��@���?�  @�(�C!H��@�
=@mp�A�
=C�                                    Bxgw�  
�          A#����@��
?xQ�@�C�����@�
=@j�HA�(�C	^�                                    Bxgw#�  �          A#�����@��
?h��@��\C������@Ϯ@g
=A�33C	@                                     Bxgw2@  �          A#�����@�\)?z�H@��C������@��H@g
=A�C
p�                                    Bxgw@�  �          A"�R��@�\?��\@��C�H��@�@eA�\)Cz�                                    BxgwO�  "          A"�R��33@�
=?k�@�{C�=��33@��
@\��A���C:�                                    Bxgw^2  �          A"�\���@�G�?aG�@��RC�����@�ff@\(�A�=qC��                                    Bxgwl�  �          A"ff��\@޸R?c�
@�G�CxR��\@�(�@Z�HA�p�C
                                    Bxgw{~  
�          A"ff��\@޸R?c�
@�  C����\@�(�@Y��A���C�                                    Bxgw�$  �          A"ff����@�p�?�R@`  C�����@�{@H��A�{C�                                    Bxgw��  "          A"�H��@�{?��@W�C�3��@�
=@G�A��HC
=                                    Bxgw�p  
�          A#\)��
=@�(�?��@�{C\)��
=@Ǯ@g
=A��C.                                    Bxgw�  �          A#
=��\)@�\?�
=@��
C����\)@���@l��A�ffC�H                                    Bxgwļ  
�          A"�R����@У�?=p�@�p�C�q����@�G�@E�A�G�CJ=                                    Bxgw�b  T          A"�R��@��?�z�@љ�C����@�  @g
=A���C�=                                    Bxgw�  "          A#
=��Q�@ۅ?.{@uC	�\��Q�@�z�@I��A��C��                                    Bxgw�  T          A"ff��ff@��
?�@>�RC	O\��ff@�ff@@  A��C8R                                    Bxgw�T  �          A"ff��{@��>��
?��
C	#���{@��@3�
A���C�                                    Bxgx�  	�          A!���
@�p�>k�?��
C�\���
@˅@.�RAz�\C33                                    Bxgx�  
Z          A"{��R@ڏ\>�?@  C	� ��R@�=q@'
=An�HC��                                    Bxgx+F  �          A"ff���@�\���Ϳ\)C�����@�z�@�RAb{C	��                                    Bxgx9�  �          A!���ff@�33>u?�{Ck���ff@���@333A�Q�C	��                                    BxgxH�  T          A!��ff@�\�Ǯ�p�Cs3��ff@�  @��AH��Cٚ                                    BxgxW8  
�          A!����
=@߮��  ���C���
=@�(�?˅A�Cc�                                    Bxgxe�  T          A!G���{@ָR��
=��\)C	�3��{@�?��@�\)C
�                                    Bxgxt�  
�          A"=q����@У׿�����HC�{����@љ�?���@�p�Cs3                                    Bxgx�*  	�          A!�����@˅���
��  C�3����@�z�?��@�p�C��                                    Bxgx��  �          A"�R����@��Ϳ�
=�G�C�����@Ϯ?u@���C#�                                    Bxgx�v  �          A!���=q@�
=�����C}q��=q@���?0��@z=qC��                                    Bxgx�  
�          A!G����
@�G����R�5Cp����
@��H>Ǯ@p�C&f                                    Bxgx��  8          A!��p�@����(���r{C�q��p�@��H���
���
CQ�                                    Bxgx�h  �          A����@����#�
�n{CE���@љ���  ����C
�3                                    Bxgx�  
�          A���p�@����6ff��G�C�3��p�@��Ϳ��E�C	.                                    Bxgx�  �          A{��z�@�z�Q���=qCu���z�@���?���AffC��                                    Bxgx�Z  �          Aff����@�zῆff����C=q����@���?�G�@���C5�                                    Bxgy   T          A�׿�>W
=A�\B�\)C)�῕�+�@��RB�k�CvW
                                    Bxgy�  
�          A(���z�?=p�A�B�
=C�=��z����@�
=B���Ce�                                    Bxgy$L  p          A���=p�@1G�@�BvQ�C��=p��#�
@�\)B�L�C4�R                                    Bxgy2�  "          A(����\@��@��B$�
C!H���\@
=q@���BU�Cu�                                    BxgyA�            Aff��  @�=q���R�S33Cu���  @��R�8Q쿙��CW
                                    BxgyP>  �          A��\)?����������C+s3��\)?˅�Q���  C(                                    Bxgy^�  
�          A	���׮@�G�?�Q�A;�
C
=�׮@j�H@J=qA�G�Cs3                                    Bxgym�  �          A\)��G�@�p�?�  A!C=q��G�@X��@6ffA�Q�C�                                    Bxgy|0  �          A�
���@{�@G�Ar=qC����@8��@_\)A�p�C�                                    Bxgy��  
�          A�R��G�@�z�@<��A�z�C���G�@Fff@�G�A�
=C8R                                    Bxgy�|  
�          A(���33@�=q@�
=B(�C�3��33@
�H@�(�BHffC��                                    Bxgy�"  �          A
�\���R@�p�@���B  CxR���R@.�R@ÅB4=qCu�                                    Bxgy��  
�          A	�����@�=q@p�Ar=qC������@�  @tz�A֣�C                                    Bxgy�n            A{�A�@�\)@�p�B�
B�3�A�@p  @�z�B_�HC ޸                                    Bxgy�  T          A��(�@�
=@]p�A��C�\��(�@�p�@���B{C�                                    Bxgy�  "          Az���33@�Q�?��A"{C}q��33@��@P  A��
C33                                    Bxgy�`  �          AG���ff@�33@tz�A��C33��ff@n�R@��BG�C�q                                    Bxgz   T          A��ƸR@�z�@G�A�(�C��ƸR@�\)@�{BG�C8R                                    Bxgz�  �          A�
��ff@�Q�@o\)A��C Q���ff@��
@���Bp�C	                                      BxgzR  
�          A=q��ff@���@�A�(�C�)��ff@w�@��
B2�CY�                                    Bxgz+�  �          A����(�@�ff?�(�@�\C33��(�@�Q�@/\)A�C�                                    Bxgz:�  "          A(���33@�\)?��@���CT{��33@�G�@5A�33C��                                    BxgzID  
�          A=q��@��?��A
=C�)��@�Q�@UA�=qC33                                    BxgzW�  
�          A\)���
@�?˅A�RCc����
@��\@Tz�A�=qC��                                    Bxgzf�  �          A�\��G�@��@)��A�{C����G�@�  @�Q�A�C�R                                    Bxgzu6  �          A\)��Q�@�Q�@z�HA��HC����Q�@�=q@��
B=qCJ=                                    Bxgz��  T          A\)��33@�=q@�HAmp�C=q��33@���@���Aݙ�C�                                    Bxgz��  
�          A  ����@�
=@9��A�(�C�����@��@�
=A��RC
p�                                    Bxgz�(  �          Ap�����@θR@`��A�p�CL�����@�p�@��RB  C�q                                    Bxgz��  �          A����
=@�Q�@>{A��C���
=@���@�
=A�33C��                                    Bxgz�t  �          A��\)@�Q�@EA�{C�=��\)@�33@��A�=qCp�                                    Bxgz�  T          A��ָR@Ӆ@8Q�A�=qCp��ָR@���@��A�33C��                                    Bxgz��  �          A�\�ٙ�@�
=@'
=AtQ�C^��ٙ�@�\)@�A�C+�                                    Bxgz�f  T          A�����
@�
=@4z�A�G�C�=���
@�p�@�(�A�C��                                    Bxgz�  �          A����
=@ۅ@H��A�{C33��
=@�{@�
=B33C
�\                                    Bxg{�  �          A  ����@��@[�A���CL�����@�G�@���B
��C.                                    Bxg{X  T          A
=���@��@P��A�p�C�{���@�=q@�(�B{C33                                    Bxg{$�  "          Az��Z=q@��@\(�A�{B����Z=q@�ff@�Q�B(�RB���                                    Bxg{3�  
�          A	��?\@�p�@�G�B7=qB��?\@X��@��B=qB���                                    Bxg{BJ  �          A���{@��
@�z�B!z�B��
��{@�@�\)BmQ�B��q                                    Bxg{P�  �          A�?�@�@��
B�RB�p�?�@��@�\Bb=qB��3                                    Bxg{_�  �          Aff@�R@���@�
=B�B���@�R@�
=@�(�BW(�B�z�                                    Bxg{n<  �          A  @b�\@�{@~{A��HB{@b�\@���@\B+ffBe=q                                    Bxg{|�  T          A?ٙ�@�33@��
A��HB�?ٙ�@��@�33B8p�B���                                    Bxg{��  T          A��@6ff@��
@��
A���B��@6ff@���@ҏ\BC
=Bq�\                                    Bxg{�.  
�          A��?��@��@��A�B�33?��@�@ڏ\BC��B�p�                                    Bxg{��  
�          A���(�@��@��A��B�uþ�(�@�=q@ָRBD  B��                                     Bxg{�z  >          A
=�\)@�@�33B�B�L;\)@�\)@�z�BW(�B�.                                    Bxg{�   T          A�H?��@��
@��RB\)B�=q?��@��H@�RBZ�B�                                    Bxg{��  
�          Az�?��
@���@��
B��B���?��
@�(�@�Bc�RB��                                     Bxg{�l  T          AQ���R@���@�  A�z�BՀ ��R@��@�(�B9p�B��                                    Bxg{�  �          AQ���\)@�p�?�G�A4��B�\��\)@�\)@���A��HB�L�                                    Bxg| �  
>          Az����\@�G�������C n���\@���?�AE�Cz�                                    Bxg|^  
�          AQ���@����(��J�HB��{��@��>��@@  B��                                    Bxg|  
�          A�H���@�33�(���ffCk����@ٙ����R� ��C��                                    Bxg|,�  
�          A����@�ff�[����C ����@�R��p���{B�33                                    Bxg|;P  p          A����@Ӆ�
�H�hz�B������@޸R<#�
=�\)B�#�                                    Bxg|I�  
�          A�����H@��ü��
��G�C�����H@�@(�AYp�CT{                                    Bxg|X�  �          AQ��ָR@���>�?Tz�C���ָR@�p�@AR=qC
��                                    Bxg|gB  �          Az���z�@�=q����n{C����z�@�\)?�@�\)Cn                                    Bxg|u�  T          AQ���{@��?(�@q�C�R��{@��R@��Ac33CW
                                    Bxg|��            Az���ff@�z�?�\)A ��C�3��ff@�33@W�A��RC��                                    Bxg|�4  >          A���ff@˅?�A'33Cc���ff@���@c�
A��
CaH                                    Bxg|��  8          A���@��?���A��C	\)��@��@Q�A�
=C�                                    Bxg|��  �          A���@�  ?�z�A'
=CE��@�
=@QG�A��
Ch�                                    Bxg|�&  
Y          A���z�@�G�?�p�A.�\C����z�@�
=@[�A�  C.                                    Bxg|��  �          A��߮@�=q@{A^�RCs3�߮@��
@tz�A�{C��                                    Bxg|�r  
�          A33���@��?ǮAC�3���@}p�@:=qA�G�C��                                    Bxg|�  T          A�\��ff@�@�
AP��CB���ff@��@c33A�33C!H                                    Bxg|��  9          A
=��R@�  @�RA|Q�CG���R@�  @xQ�A�C�R                                    Bxg}d            A\)��=q@��\@X��A�ffC����=q@���@�33B �CaH                                    Bxg}
  T          AQ����@�?�@���B��3���@���@[�A�=qB�                                    Bxg}%�  
�          A����@�\)@
=Am�C
=���@�  @z=qAȸRCQ�                                    Bxg}4V  
�          A�����@���@��AaG�C
=���@��
@p��A��RC!H                                    Bxg}B�  
?          A����33@�=q@O\)A�ffC�{��33@��\@���A��C��                                    Bxg}Q�  T          A���θR@��@UA�{C

=�θR@���@�B   C�3                                    Bxg}`H  �          A�����@˅@=p�A�33C�����@�p�@��A�  C��                                    Bxg}n�  "          Az���@�{@#33A
=C���@�z�@�G�A�(�C^�                                    Bxg}}�  "          A����p�@�(�?�33@�33C
��p�@�\)@E�A���C	
=                                    Bxg}�:  
�          A������@ٙ�?�  @�  C�����@Å@N{A�
=C#�                                    Bxg}��  "          Az���@��H?��@�G�CT{��@�ff@C33A��C	@                                     Bxg}��  "          AQ����
@�?Q�@��HC�����
@�z�@1G�A�G�C{                                    Bxg}�,  
�          A���Å@�ff?
=q@VffCW
�Å@�
=@%�A�33CY�                                    Bxg}��  �          A����z�@�
=>�ff@333CW
��z�@���@   Ayp�C:�                                    Bxg}�x  
�          A���G�@ۅ>aG�?���C����G�@Ϯ@\)A^ffC�                                    Bxg}�  T          A  ���H@�{���?\)CJ=���H@أ�?�=qA�C��                                    Bxg}��  �          A�R���H@�33�u��G�C�����H@��
?��
A3�C�f                                    Bxg~j  
Z          A
=��\)@޸R=���?&ffC����\)@��
@��AX  C�                                    Bxg~  �          A
=����@�
=���Q�C@ ����@�=q?��A33C޸                                    Bxg~�  "          A\)�\@��
��  ���C��\@�z�?�\A2{Cz�                                    Bxg~-\  �          A���Q�@�=q?��RAG�CaH��Q�@��H@Z�HA�=qC�H                                    Bxg~<  �          A�����@�
=?��\@�(�C�����@ə�@P  A��
C�\                                    Bxg~J�  �          A��=q@�\)@�Aa�C����=q@���@��RA�p�C�                                    Bxg~YN  
�          A�H���@�{@z�Adz�Cp����@�
=@�\)A��
C��                                    Bxg~g�  
�          Aff��
=@�ff?��@�\)C����
=@��H@AG�A�Q�Ck�                                    Bxg~v�  
�          A�H�У�@��=�G�?&ffC�q�У�@Ϯ@z�AJ�RC�                                    Bxg~�@  T          A
=��
=@�녿O\)���C����
=@У�?�\)@ٙ�C޸                                    Bxg~��  
          A�H��(�@˅��ff��p�C	=q��(�@���?Q�@�\)C	{                                    Bxg~��            A�H��z�@��ÿ��
�HC	�f��z�@�p�>�G�@*=qC	                                      Bxg~�2  �          A�����@�ff?+�@��\C������@θR@(Q�A�33C��                                    Bxg~��  "          A  �˅@�{?�  @�=qC� �˅@�G�@K�A�z�CL�                                    Bxg~�~  �          A(���z�@���?��A�C�{��z�@�\)@P  A�{C�q                                    Bxg~�$  
�          AQ���(�@߮?h��@���CaH��(�@�{@7
=A�=qC��                                    Bxg~��  �          Az���Q�@�p�?�@J=qC:���Q�@Ϯ@{Apz�C{                                    Bxg~�p  �          A�
��ff@ָR>\)?Y��C���ff@���@�\AF=qCL�                                    Bxg	  
�          A  ����@���<#�
=�Q�C������@�(�?��A7�
C�q                                    Bxg�  T          A(����@�z�u��Q�C�q���@�z�?�A/�
Cٚ                                    Bxg&b  �          AQ��ۅ@�33���8Q�C#��ۅ@˅?޸RA(z�C	(�                                    Bxg5  "          Az���33@�(��L�;��
C����33@�(�?�A/
=C	�                                    BxgC�  "          A�����@ָR���Ϳ(�Cs3���@�
=?��
A+�C}q                                    BxgRT  �          A���=q@ָR�u��z�Cs3��=q@�Q�?�33A�RCQ�                                    Bxg`�  �          Ap���p�@��
�B�\��\)CJ=��p�@��?�A (�C	5�                                    Bxgo�  �          Ap��ᙚ@Ϯ���Ϳ(�C	Y��ᙚ@ȣ�?��HA$(�C
^�                                    Bxg~F  
�          AQ���(�@�녾8Q쿈��C
����(�@Å?�=qA��Ck�                                    Bxg��  
�          A(���@��>��R?�C
k���@�
=@�\AF=qC�3                                    Bxg��  "          A�
��@\?G�@��C  ��@��
@�AmC+�                                    Bxg�8  "          A����33@�(�>�@1G�C����33@�  @G�A\Q�C	��                                    Bxg��  
Z          AG��׮@ٙ�=�\)>��C���׮@У�?�Q�A:�HC�3                                    BxgǄ  T          A���ٙ�@�Q�>�{@�
C&f�ٙ�@��@(�AR�HC��                                    Bxg�*  	�          A�����H@�
=>\)?W
=C�����H@�?�p�A>=qC��                                    Bxg��  
Z          A=q��G�@ҏ\>B�\?�{C���G�@���?�p�A=p�C
G�                                    Bxg�v  �          AG���33@��\)�O\)C����33@ָR?�G�A)p�C�\                                    Bxg�  �          A����33@������aG�C���33@�{?޸RA'�
C�)                                    Bxg��  T          A����33@����Q�
=qC����33@�?��A,��C�                                    Bxg�h  �          A����=q@�{=u>\Cn��=q@��?�Q�A;�C�
                                    Bxg�.            A{��=q@�G�=u>�p�C
=��=q@�Q�?��HA<(�C33                                    Bxg�<�  =          A�\���@�R>�\)?��C�����@��
@p�AT  C�                                    Bxg�KZ  	�          A�\���
@�Q�>aG�?��
CG����
@�@
=qAO
=C�
                                    Bxg�Z   o          A{����@�ff>�?G�C�H����@���@�
AEC�
                                    Bxg�h�  "          A�����@��
=��
>�C�����@�\@33AE�C�                                    Bxg�wL  	�          A��ȣ�@陚>��R?�{C���ȣ�@�ff@  AX��C\                                    Bxg���            AG���=q@�>�33@�CaH��=q@�=q@  AYC��                                    Bxg���  T          Az��ə�@�z�>\@�
Ck��ə�@���@G�A\��C�f                                    Bxg��>  
�          Az���33@�=q?(�@k�C���33@��@{Apz�C��                                    Bxg���  
�          AQ���z�@�
=?s33@�Q�C���z�@θR@1G�A��C��                                    Bxg���  
�          A  �Ӆ@�ff?��R@�C���Ӆ@�(�@>{A��C	&f                                    Bxg��0  T          A(���(�@�\)?��@ə�C�\��(�@�
=@2�\A�Q�C�
                                    Bxg���  T          AQ����@�ff?�{@�ffC�R���@�@1G�A��C^�                                    Bxg��|  �          A(��ҏ\@���?���@θRC(��ҏ\@�Q�@4z�A��
Cp�                                    Bxg��"  
�          A�����
@�{>��?k�C�����
@���?�(�A=��C�)                                    Bxg�	�  
�          A{���@�{>W
=?��
C�
���@�z�@�ABffC�                                    Bxg�n  
u          A����@�G�>�  ?\C޸����@�\)@AH��C+�                                    Bxg�'  �          Ap���G�@�  >�33@�C{��G�@��@
=qAP��Cz�                                    Bxg�5�  
?          A���θR@�(�?�  A(�C(��θR@�Q�@O\)A�z�C�f                                    Bxg�D`  o          Ap���
=@�?�  @�Q�C����
=@Ӆ@E�A�CE                                    Bxg�S  �          Ap���(�@�ff?\ACff��(�@��@U�A��HC�                                    Bxg�a�  �          A{����@�Q�?�
=A��CG�����@�z�@P��A��\C��                                    Bxg�pR  �          A{�ə�@�\)?(�@i��C
�ə�@ڏ\@p�Am�C�R                                    Bxg�~�  �          A{�ȣ�@��?#�
@s33C�q�ȣ�@ۅ@\)Ap(�Cc�                                    Bxg���  �          A=q����@�?@  @�ffC �\����@�G�@)��A33C��                                    Bxg��D  T          A=q��=q@�ff?(��@z�HC0���=q@���@#33Au��C��                                    Bxg���  �          Aff�\@�R?.{@�G�C.�\@�G�@$z�Aw
=C��                                    Bxg���  "          A=q�Ǯ@�\>�Q�@�Ch��Ǯ@�  @{AT��C�q                                    Bxg��6  �          A{�ʏ\@�>��
?��C!H�ʏ\@�p�@	��AN�HCk�                                    Bxg���  T          A�\���
@�p�?0��@��
C� ���
@�  @#�
AvffC&f                                    Bxg��  �          A����=q@�
=>�?@  C0���=q@�ff?��HA<(�CE                                    Bxg��(  �          A���=q@�Q�=�G�?&ffC&f��=q@�  ?��A4��C8R                                    Bxg��  T          A����
@�ff=�Q�?�C����
@�{?�z�A733C��                                    Bxg�t  "          A�ȣ�@�Q�>�=q?У�Cٚ�ȣ�@�ff@ffAI�C�                                    Bxg�   T          A��(�@�=���?(�C����(�@�p�?�z�A7�C��                                    Bxg�.�  �          AG���(�@�z�=L��>���C���(�@���?���A1C                                    Bxg�=f  
�          Ap����@��H>�?J=qC���@�=q?�(�A=�C{                                    Bxg�L  
Z          A�����H@��H?(��@~�RC����H@�{@�RAp��CE                                    Bxg�Z�  T          A������@��
>.{?��\Cc�����@��H@ ��ABffCxR                                    Bxg�iX  
�          A������@�R�u��33C�����@߮?޸RA((�C��                                    Bxg�w�  T          A�����@�z���B�\C� ���@�{?�z�A (�CG�                                    Bxg���  T          AQ���33@�G�>��R?��C�3��33@�\)@
=AMp�C+�                                    Bxg��J  "          Az���ff@�p�>�p�@��C ����ff@�33@��AV=qC�R                                    Bxg���  
�          A����R@��
?fff@��B��3���R@��@0��A�=qC �{                                    Bxg���  �          A�R���\@�?��@���B������\@��H@>�RA�{Cs3                                    Bxg��<  "          A�R��@�ff?�z�@���B��
��@�@AG�A�(�C ^�                                    Bxg���  
(          A�
��(�A=q?�33A(�B�ff��(�@�@W
=A��B�aH                                    Bxg�ވ  "          A(���z�A33?�Q�@�Q�B�.��z�@���@J�HA��B�                                    Bxg��.  �          A���Q�A
=?�33AG�B�����Q�@��H@W�A���B��)                                    Bxg���  �          A\)��Q�@�?��RA��B��R��Q�@�\@Tz�A�  C�                                    Bxg�
z  
�          A���=q@���?�\)A{C^���=q@���@XQ�A��Cٚ                                    Bxg�   �          A�
��z�@���?У�AB�����z�@�z�@^{A�
=C Q�                                    Bxg�'�  
?          Az���G�A�?У�AG�B�u���G�@�@b�\A��
B��)                                    Bxg�6l            AQ�����A\)@�
AZffB�\)����@���@�\)A�G�B���                                    Bxg�E  "          AQ���\)A(�?�p�A;�B��\)@���@z�HA�=qB�=q                                    Bxg�S�  "          AG���@�=u>�p�B�W
��@�(�?�A6�\C ��                                    Bxg�b^  "          A��p  @�ff@�A�{B�\�p  @�p�@�  A�(�B�{                                    Bxg�q  T          A���H@��@��BG�B�aH��H@���@�
=BBz�B�u�                                    Bxg��  �          A���U�@���@�{B (�B���U�@�  @�Q�B1��B��                                    Bxg��P  �          A��S33@��@��RBG�B���S33@���@�{B@�B�                                    Bxg���  T          A��� ��@�=q@���B  B�
=� ��@���@�\BE�B�q                                    Bxg���  T          A33�y��@��R@W
=A�(�B���y��@�@��B��B��=                                    Bxg��B  T          A=q���
@�\)?ٙ�A#
=B�ff���
@��H@c�
A��B���                                    Bxg���  �          A�����@�33?��HAz�B�\����@��@R�\A��RB�(�                                    Bxg�׎  �          Ap���G�@�=q?L��@��B�#���G�@��@(��A��B�{                                    Bxg��4  	�          AG�����@��@"�\Av�RB�������@��@��A�{B�                                      Bxg���  T          AG���33AG�@�AS�B�Q���33@�\@�G�A�{B�8R                                    Bxg��  T          A{��p�@��H@�AB�HB��
��p�@�z�@u�A���B��H                                    Bxg�&  �          A�\��p�@�
=?���@�
=B�����p�@�\)@:�HA���C )                                    Bxg� �  "          Aff���@�(�>u?�33B�
=���@�@�AB�RC ��                                    Bxg�/r  
�          A=q����@��>�{@�B�����@�{@p�AU��B���                                    Bxg�>  �          A{����@�=q�z��_\)B�.����@��?��\@�{B�                                    Bxg�L�  
�          Az���{Ap��k���\)B�����{@��?�
=A#
=B��                                    Bxg�[d  �          A�����
A{?��A\)B�=q���
@��@`  A�
=B�(�                                    Bxg�j
  "          A33����A=q?�  A,��B������@�Q�@g
=A�33B�{                                    Bxg�x�  
(          A  ��33@�z�@33Amp�B� ��33@�z�@���A���B�\                                    Bxg��V  �          A��
=AG�>�
=@\)B���
=A   @ffAb�RB�                                    Bxg���  �          A����
AQ�>�
=@!G�B�L����
A
=@��Ag
=B�.                                    Bxg���  
�          A���`  A�\�����A��B�(��`  A
=q=L��>���B��                                    Bxg��H  �          Az��\)A���G
=���B��H��\)AQ�}p���Q�B���                                    Bxg���  �          AQ���
A���@  ���
Bǣ׿��
A�Ϳ^�R����BƳ3                                    Bxg�Д  �          A�H����A33�XQ���{B�����A�
���
���B���                                    Bxg��:            A�R�$z�A��qG����HB֞��$z�A{��\�-G�B�8R                                    Bxg���            A�R�5�@�\)��=q��ffB�#��5�A	G��*�H��B�u�                                    Bxg���  �          Ap��c33@����p���33B�=�c33Az����hz�B�k�                                    Bxg�,  "          A33�Tz�@�\)�������HB�G��Tz�A�\�G
=��p�B�=q                                    Bxg��  �          A����@�p���z����Bٽq���AQ��vff��=qB�                                    Bxg�(x  �          A{���R@ۅ�����!(�B�Q���RA ����=q��{Bϳ3                                    Bxg�7  �          AQ���{@�  @�G�A�B�k���{@��
@���B{B���                                    Bxg�E�  T          AG���=q@��H@QG�A���B��3��=q@�(�@�
=A�B�.                                    Bxg�Tj            A  ��p�@�G�@�p�A��B�\��p�@��@�33Bp�C :�                                    Bxg�c  <          A������@׮@��RA�33B�����@�@˅B'�C�                                    Bxg�q�  �          Az����H@��@E�A�
=B�
=���H@У�@�A��
C                                     Bxg��\  �          A�\���
@��
�-p���z�B������
@�G��xQ���ffB��                                    Bxg��  n          A�
��=qA��	���S
=B����=qA  �k���
=B�(�                                    Bxg���  �          A������@�{����K�B�������@�{��=q����B���                                    Bxg��N  �          A  ��p�@�33?(��@\)C�q��p�@���?��HA?33C0�                                    Bxg���  �          A(���  @�p�?��
AG�Cn��  @��@;�A���C
�=                                    Bxg�ɚ  
�          A����  @�\)?z�H@�C����  @��@\)AZ�HCn                                    Bxg��@  T          A�
���@�ff?��@ӅC�{���@�G�@ ��Av=qC
s3                                    Bxg���  �          A�
��(�@޸R?��
A.ffC^���(�@�(�@S33A���C�
                                    Bxg���  
�          A33����@�(�?�\A-�C�H����@љ�@Tz�A���C                                    Bxg�2  T          A����G�@���?333@�ffC����G�@�=q@
�HAR=qC	�                                    Bxg��  
�          A�\��33@�G�?���@��C���33@��
@'
=A��C��                                    Bxg�!~  
�          A{����@�  ?:�H@�Q�B�������@���@��Al��B�u�                                    Bxg�0$  �          A���@�33?B�\@��C��@�Q�@z�Ac�Cs3                                    Bxg�>�  
Z          A����G�@��
�#�
��Q�CO\��G�@�{?�\)A��C�                                    Bxg�Mp  �          A����
=@ᙚ���R��Q�Ck���
=@���>�33@��C                                    Bxg�\  �          A�H��  @�(���{���C@ ��  @�Q�>L��?�p�C�R                                    Bxg�j�  
�          A�R��33@��>�{@�C�q��33@أ�?��A:�\C�                                    Bxg�yb  
�          A�H����@�\?E�@�p�C������@׮@�
Ac\)CaH                                    Bxg��  �          A�
�ָR@љ�?�(�A{C���ָR@�=q@7�A�
=C	޸                                    Bxg���  T          A����  @��@��AT��C=q��  @��@Q�A�33Cn                                    Bxg��T  �          A���p�@�33?\)@Z�HC� ��p�@��\?�A3
=C
=                                    Bxg���  l          A����Q�@���=�Q�?\)C	���Q�@�33?��
A�\C	�{                                    Bxg�   �          A����p�@��ÿ��H���C���p�@�(�>�  ?�(�C8R                                    Bxg��F  
�          A33�陚@��Ϳ&ff�}p�C{�陚@���?(��@�Q�C{                                    Bxg���  
�          Az���@��ÿ0����(�C
����@���?333@��C
�{                                    Bxg��  "          A����R@Å������G�C�3��R@ƸR>u?�C@                                     Bxg��8  T          A  ��\)@�
=������RCz���\)@��
�#�
���
CǮ                                    Bxg��  T          A(���Q�@��H������\CW
��Q�@���#�
�uC�H                                    Bxg��  :          Az����@�Q��$z��{\)C�f���@�ff��p��
=CT{                                    Bxg�)*  
          A����ff@љ���\)�4Q�C�f��ff@ٙ��������C�{                                    Bxg�7�  
�          A=q��p�@ə����[
=C	����p�@�z�\(����C+�                                    Bxg�Fv  
r          A�
����@�
=�0  ��G�CE����@�z῏\)���HB�\)                                    Bxg�U  
�          A(���G�@���<����\)C����G�@���{� ��C ��                                    Bxg�c�  T          A����p�@�p��HQ����C����p�@�{��\)�Q�C�                                    Bxg�rh  
�          A����33@ҏ\�A���Q�C���33@�\������C��                                    Bxg��  
�          A��  @��H�8����Q�C����  @ᙚ��33��RC�R                                    Bxg���  T          A=q����@ȣ��e��\)C������@����	���J�\C�3                                    Bxg��Z  
�          A����@�ff���\���C�{���@����������C�                                    Bxg��   
�          A�H��p�@������R�	p�C�{��p�@�
=��z���C��                                    Bxg���  �          A���@�p���\)���HC�f��@�=q�dz���{C                                      Bxg��L  T          A33��@�(��������Cٚ��@�(�����ȸRC0�                                    Bxg���  �          A=q����@��H��z��33C������@��H������=qC�                                    Bxg��  �          A���(�@}p����� \)C����(�@��R������Q�C+�                                    Bxg��>  T          AG����@e������\)C�q���@�(��������HC�H                                    Bxg��  
�          A�����@5���\)��Cp�����@{���  ��C5�                                    Bxg��  �          A�R��R@333���
��C� ��R@x��������RC�                                    Bxg�"0  �          A�����@'�����33C�=���@o\)���
��C�                                    Bxg�0�  
�          A  ����@����  �{C!:�����@`  �����{C��                                    Bxg�?|  
�          Aff�Ӆ?�G���=q�&=qC%{�Ӆ@@���������C�                                    Bxg�N"  �          A��\)@�������C ޸��\)@^�R��G��z�C��                                    Bxg�\�  T          A�
��p�?�{��ff�/��C#B���p�@HQ���p��Q�C)                                    Bxg�kn  T          A  �Ǯ@(����=q�"�C\�Ǯ@tz���(��33C��                                    Bxg�z  
�          A(���{@(����Q��'  C���{@vff����Q�C!H                                    Bxg���  "          A�����H@#�
����z�C  ���H@l(���{��HC��                                    Bxg��`  �          A�\��33@QG���z���C� ��33@�p��vff���C��                                    Bxg��  T          A���
@L����ff��z�C����
@����[���Q�C��                                    Bxg���  �          A=q��\)@e��z���ffC�q��\)@�z��R�\��  Cٚ                                    Bxg��R  �          Aff��(�@p���N{��C��(�@�33�ff�u��Cz�                                    Bxg���  
Z          A��陚@g
=�Vff��
=C��陚@���!G����HC�H                                    Bxg���  T          AG���p�@tz��\(���p�C���p�@��R�#�
����C�                                    Bxg��D  
�          AG���33@��\�Q���z�C)��33@��ff�v�HC��                                    Bxg���  
�          A������@����c�
����C&f����@�{�(Q���ffCG�                                    Bxg��  "          A{��(�@qG��n�R��z�C(���(�@��R�6ff���C�3                                    Bxg�6  
Z          Ap���33@�G��N{��33Cs3��33@��
����f�RC=q                                    Bxg�)�  
�          A����\)@��H��p����HC8R��\)@���J�H���
C�q                                    Bxg�8�  �          A��  @���������C� ��  @��o\)��C8R                                    Bxg�G(  �          A{��p�@��\)�֏\C ����p�@�(��'
=��p�B�(�                                    Bxg�U�  �          A�
���
@�G��vff��
=C����
@޸R����}G�B��                                    Bxg�dt  �          Ap���33@fff��z��G�C���33@��H�������CQ�                                    Bxg�s  �          A�H���þ#�
��\)�[{C5�3����?�z���33�Up�C%
                                    Bxg���  �          A���z�>\��=q�VG�C/����z�?�33��\�L\)C ��                                    Bxg��f  
�          A����\?333�陚�V��C,�=���\@����  �J{C��                                    Bxg��  "          A�R��=��
��p��g  C2�q��?�\)���^p�C ޸                                    Bxg���  
�          A�x��=������\)C2}q�x��?�  ����tp�C�\                                    Bxg��X  
�          A�\���R>������H�v�HC/�����R?���33�jCp�                                    Bxg���  �          A��aG��fff� (�.CBB��aG�?J=q� Q��=C'h�                                    Bxg�٤  "          Az��S33������ffu�CM�\�S33=�Q����C2�                                     Bxg��J  	�          A
�H�g
=�}p�����CC\)�g
=?&ff���H�HC)��                                    Bxg���  :          A(���z�?�����p��X�C'�)��z�@�R��G��H=qC\                                    Bxg��  
          A����R?fff����r��C'�����R@{����a  C��                                    Bxg�<  
�          A=q��
=?����Q��h�C#����
=@5���=q�TC��                                    Bxg�"�  T          A{�s33?�z����H�|\)C#
=�s33@0�����fC{                                    Bxg�1�  �          A�ÿ��;��
=¤G�CK������?�=q�	��W
C�f                                    Bxg�@.  �          A33�����\)�  33C6Q���?�
=�G��C�q                                    Bxg�N�  
�          A
�\��(�?(���W
C"� ��(�@�� ����C=q                                    Bxg�]z  
�          A(���>��R��H�\C,(���?��R��{8RC
�{                                    Bxg�l   	�          A	�l(�?����33�|�C#��l(�@'
=��
=�g�RC�                                    Bxg�z�  "          A	p��~{�����H�yffC4}q�~{?��
���p�RC�                                    Bxg��l  "          A���W
=?�R�����C)���W
=@p���  �w=qC�                                    Bxg��  
�          Az��=p�?�{����ffC^��=p�@-p���  �w\)C	n                                    Bxg���  
�          A���]p�?333����.C(� �]p�@�\��
=�s��C�                                     Bxg��^  
�          A���7���\)� z��3C5O\�7�?˅��(���C                                    Bxg��  "          A�
�   �����p�aHC;���   ?������fC�\                                    Bxg�Ҫ  �          A�\�@  >������RC.T{�@  ?�{��(��fCJ=                                    Bxg��P  T          A�
�[�?��
������C#W
�[�@%�����m�
C�                                    Bxg���  "          A=q�Dz�?�\����u�C*xR�Dz�@ff��Q��~�C�\                                    Bxg���  
�          Ap���\)?(����\)C �H��\)@33����(�C#�                                    Bxg�B  
�          A{��{>�z��33p�C)����{?�(���
=�{C=q                                    Bxg��  �          AG���p���G���.CB^���p�?�p�� ��aHC��                                    Bxg�*�  	�          A�R���\?�ff���\�Ez�C$����\@{��{�4Q�C�                                    Bxg�94  T          A (���{@Dzῼ(��*�\C޸��{@S�
�O\)��33CJ=                                    Bxg�G�  �          @���z�@#33��G��33C ����z�@0  �8Q���{C��                                    Bxg�V�  "          @�{���?�
=���X  C)8R���?��
��G��0z�C&��                                    Bxg�e&  
�          A�R��33?�����ff�/�
C)����33?�p���(��
{C'�\                                    Bxg�s�  	�          A=q��@(�ÿ�G��HQ�C ����@<�Ϳ�Q��{C�                                    Bxg��r  n          A����
@G
=���s�
Ch����
@^�R��Q��$Q�C�R                                    Bxg��  	�          @�����p�@(Q���a��C�)��p�@>{������C��                                    Bxg���  l          @��R��?��׿�=q�p(�C(J=��?�p���G��E�C%}q                                    Bxg��d  
          A z�������\�У��;�
C=T{����h�ÿ����V=qC:��                                    Bxg��
  
(          A{���H���R� ������C=B����H�333�-p�����C9B�                                    Bxg�˰  �          A���
=��p��;����HC6����
=>u�<(����C2+�                                    Bxg��V  
�          A������(��� ����ffC8��������%��G�C4�R                                    Bxg���  
�          A���(��
=q�>�R���\C8
��(�=��
�A���p�C3k�                                    Bxg���  T          Aff����.{�+����
C9�������0�����RC4޸                                    Bxg�H  
Z          A=q��
=>.{����=qC2�\��
=>�ff��=q�
=C0��                                    Bxg��  �          Ap���Q�Y���(���G�C:z���Q쾮{�$z�����C6�\                                    Bxg�#�  
�          A=q���H����Q����CD�����H���i����(�C?\)                                    Bxg�2:  
�          A ����{����0  ����CE}q��{��33�J=q��p�C@��                                    Bxg�@�  
Z          A=q����Q��(��z�HC@�=�����H�\)��33C=
=                                    Bxg�O�  "          A��G�=����.{���\C3E��G�?&ff�(�����C/�                                    Bxg�^,  
�          A����ff?�  ���H�(��C(����ff?�\��\)���C'{                                    Bxg�l�  �          A �����?xQ�u��C-
���?u>\)?uC-#�                                    Bxg�{x  
Z          A���\)?��H<#�
=��
C+\)��\)?�
=>��?�C+�\                                    Bxg��  "          A ����(�?��;L�Ϳ���C(�\��(�?���>�?fffC(�                                     Bxg���  
�          Ap����H?������R�
�HC&����H?�(�=���?5C%�H                                    Bxg��j  T          A���\)?�����l��C*}q��\)?��u��Q�C)�                                    Bxg��  
�          A� Q�?Q녿J=q���HC.+�� Q�?u�(���G�C-&f                                    Bxg�Ķ  
�          Ap��   ?@  �\(��ÅC.���   ?h�ÿ0����(�C-z�                                    Bxg��\  �          A�� z�?�R�����z�C/��� z�?Q녿fff�ʏ\C..                                    Bxg��  
�          A=q��>8Q�Tz���(�C2�q��>�{�G���  C1�\                                    Bxg��  �          A{�����E���C4޸��=#�
�G����C3�q                                    Bxg��N  �          A�H� �ÿzῚ�H�(�C8
� �þ��
��ff��RC6@                                     Bxg��  
Z          A�\� zᾣ�
��ff��C6J=� z�L�Ϳ����
C4W
                                    Bxg��  �          A�� Q�0�׿�����C8�H� Q�Ǯ��  �)G�C6                                    Bxg�+@  
�          A���33�Y�����
�/33C:&f��33�����?�C7                                    Bxg�9�  
�          A�R��\�.{��Q�C6����k��=p����RC5��                                    Bxg�H�  
Z          A�\�{��Q쾅�����C4���{����=q��
=C4:�                                    Bxg�W2  T          A33��z�?�(��G���
=C%���z�@��Q�� ��C%#�                                    Bxg�e�  
�          A(����@	���z�H���HC$���@�
���g�C#�R                                    Bxg�t~  T          A�
��?�ff?@  @�  C'.��?�{?���@�  C(�                                     Bxg��$  �          A(�� (�?�{�aG��\C&�� (�?�\)>.{?�C&�H                                    Bxg���  
Z          A�� ��?\>��?�=qC)O\� ��?��H>�@O\)C)�                                     Bxg��p  "          Az��p�?У�>\@*=qC(�
�p�?\?5@�C)^�                                    Bxg��  "          A�
��H?J=q=�G�?E�C.s3��H?B�\>�=q?��C.��                                    Bxg���  "          A��p�?�����R�
=qC'�=�p�?�=u>�
=C'^�                                    Bxg��b  �          A��ff?�Q�>�z�@�C)�3�ff?���?z�@�G�C*��                                    Bxg��  "          A��?&ff��Q���
C/}q��?aG������z�C-��                                    Bxg��  
�          A�\��
?�
=�aG���  C*0���
?�=q���w�C).                                    Bxg��T  �          A����\?�\)����s33C(���\?ٙ��W
=��
=C(=q                                    Bxg��  T          A����
?�\)���
��C*����
?��>��?�C*�q                                    Bxg��  T          Aff��?��H���
��C(:���?�Q�>�=q?���C(\)                                    Bxg�$F  �          A��� ��?�
==�Q�?&ffC&� � ��?�\)>��H@X��C&��                                    Bxg�2�  �          A��� Q�@��Q���RC%h�� Q�@�=��
?
=qC%5�                                    Bxg�A�  "          A�� ��?�녾u��Q�C(p�� ��?�z�=���?5C(W
                                    Bxg�P8  T          A=q��
=?Ǯ��  ��  C(�f��
=?�=q=��
?\)C(Ǯ                                    Bxg�^�  �          Aff��33@	��>aG�?\C$����33@�
?&ff@�33C%T{                                    Bxg�m�  T          A�\����@:�H?k�@ϮC�{����@*�H?�  A*{C ��                                    Bxg�|*  �          A��
=@@  ���
�33C
��
=@@  >���@
=qC{                                    Bxg���  
(          A (���@	�����U�C$W
��@�ͼ#�
��\)C$                                      Bxg��v  
�          A���=q@-p�>�Q�@"�\C L���=q@%�?h��@�\)C!0�                                    Bxg��  �          A ����Q�@��?�{@�ffC���Q�@z=q?��HAa�C�                                    Bxg���  
�          A����
@�
=?}p�@�G�C�3���
@�p�?��AYG�CǮ                                    Bxg��h  
�          A �����@U���8Q�Cp����@QG�?+�@�  C޸                                    Bxg��  �          A��
=@��?=p�@�  C�=��
=@tz�?ǮA2=qCL�                                    Bxg��  "          AG����
@p  ?5@���C=q���
@aG�?�(�A'�C��                                    Bxg��Z  "          AG��߮@w�?���@�C��߮@dz�?���AS�
C��                                    Bxg�    
�          A�\���@p  ?n{@ҏ\CY����@^�R?�Q�A?�C�                                    Bxg��  m          A��R@>�R>u?�p�C@ ��R@7
=?Y��@ÅC                                    Bxg�L  �          A �����H@QG�>��R@\)C�q���H@HQ�?}p�@ᙚC�H                                    Bxg�+�  "          A �����@1G�?��AX  C8R���@z�@=qA���C"^�                                    Bxg�:�  
�          A ����33@'
=@G�Ahz�C p���33@Q�@!G�A�C#��                                    Bxg�I>  T          AG���
=@'
=?ǮA2�\C � ��
=@�R@z�Am�C#ff                                    Bxg�W�  T          AG���G�@#�
?���A�C!@ ��G�@{?��AW\)C#��                                    Bxg�f�  
�          A �����H@{?Tz�@�ffC!�3���H@  ?�=qAz�C#�                                    Bxg�u0  
�          A{��p�@33>���@��C#O\��p�@(�?E�@���C$
                                    Bxg���  "          AG���{?�=q�Ǯ�2�RC&����{@������\C$�)                                    Bxg��|  
�          A���\?\�0  ����C(�H��\@�
����ffC$�                                    Bxg��"  
s          AQ���@ ���{�z{C%Y���@���  �DQ�C"u�                                    Bxg���  
�          Az���G�@���@�����C!ٚ��G�@>{�(���C޸                                    Bxg��n  T          A(���z�@*=q�I�����C����z�@P���!G����\Cp�                                    Bxg��  T          A���p�@A��L(���33CQ���p�@hQ���R���CG�                                    Bxg�ۺ  �          A�H��G�@W
=�{��z�C}q��G�@s33���H�@��C�H                                    Bxg��`  �          A\)��\@Tz���R����C�
��\@qG���p��B�\C��                                    Bxg��  
�          A\)��Q�@`���
=���Cp���Q�@z�H�����0��CǮ                                    Bxg��  �          A�R��G�@fff������HC����G�@s�
�������C��                                    Bxg�R  
�          A=q��=q@b�\>B�\?��C5���=q@Z�H?n{@��C�R                                    Bxg�$�  T          A���  @\��>�33@p�C�\��  @S33?��@�33C�=                                    Bxg�3�  �          A z���G�@j�H?�=q@�
=Cs3��G�@W
=?�=qAR�RCxR                                    Bxg�BD  �          A ����G�@S33�k���z�C�)��G�@Q�>��@W
=CǮ                                    Bxg�P�  �          A ����G�@Y�����Q�C���G�@Z�H>�=q?��C�f                                    Bxg�_�  
�          A z���Q�@u���ff�Mp�C\)��Q�@u>�p�@)��CO\                                    Bxg�n6  "          A z���{@�
=������C��{@���8Q쿡G�C�                                    Bxg�|�  T          A���{@��׿�33�(�Cs3��{@�p��aG��˅C��                                    Bxg���  �          A����(�@�
=�z��m��C�q��(�@�녿�������C                                    Bxg��(  
�          A����(�@�G��8�����C����(�@�녿�(��aG�Cp�                                    Bxg���  T          A33��Q�@�\)�(����C���Q�@��Ϳ��R�(��Cz�                                    Bxg��t  �          A{��  @�\)=�\)?
=qC:���  @��?��\@��HC�3                                    Bxg��  �          A (�����@�
=?��
AMC
  ����@�
=@6ffA��HC�
                                    Bxg���  �          Ap���Q�@�{@8��A�{C�)��Q�@~{@tz�A�G�Cu�                                    Bxg��f  �          A����@��H>���@1G�C}q���@��
?У�A7�C�{                                    Bxg��  T          A�H���\@��>�?c�
C޸���\@�  ?�=qA�RC�                                    Bxg� �  	�          A����@�p��h����Q�C���@�\)>��
@\)C�R                                    Bxg�X  "          A �����@�G���ff�1�C�\���@�\)��\)��p�Cٚ                                    Bxg��  
�          A (�����@�z��   �h  C:�����@�{�@  ���
C                                    Bxg�,�  �          A �����H@��\�)������C����H@��׿�Q��&�\C�q                                    Bxg�;J  �          A   ��\)@�G���
=�A�C޸��\)@��ÿ\)��  C
�=                                    Bxg�I�  "          A���ʏ\@�
=�Tz����C޸�ʏ\@���>��?���C�                                    Bxg�X�  
Z          A�
��ff@���G��)C���ff@�(���Q��"�\C                                    Bxg�g<  	          A  ����@�Q�����L��Cp�����@��ÿ333����C�                                    Bxg�u�  	          A�
����@�����v{C������@��R���
��C	�                                    Bxg���  	          A���\)@��׿�33�U��C)��\)@�=q�E����
C�                                    Bxg��.  �          A�R���@��\�\)�xQ�Cc����@�\)?���@�C�                                    Bxg���  �          A�R��Q�@���z���{C���Q�@������\)C�                                    Bxg��z  �          A�R��(�@���9����
=C)��(�@�����=p�C��                                    Bxg��   T          A��G�@�������  C�f��G�@��ÿ����%p�CQ�                                    Bxg���  T          AG�����@p  ��\)�ffCxR����@}p����qG�C&f                                    Bxg��l  �          A�R��Q�@�Q��{��z�CB���Q�@�(���Q��(�C\                                    Bxg��  
�          A��\)@�(��  ��G�Ck���\)@�Q쿘Q��\)C=q                                    Bxg���  "          A�\��ff@�z�����s�CQ���ff@�  ��Q��=qC�                                    Bxg�^  T          A(���\)@hQ���:�HCO\��\)@z=q�W
=���\C�{                                    Bxg�  �          A�\�陚@p�׿�p��>�RC� �陚@����^�R���RC�q                                    Bxg�%�  �          A33���H@p�׿�=q�G�
C�{���H@�=q�u��Q�C��                                    Bxg�4P  �          AG���@l(����f�HC����@�녿�(���\C�=                                    Bxg�B�  
(          AQ���@����HQ����
C&f��@�\)����pQ�C��                                    Bxg�Q�  "          A����@l���.{����C����@��R���LQ�C�q                                    Bxg�`B  �          A����(�@Tz��&ff���\C���(�@s�
���I�C�H                                    Bxg�n�  "          A��\)@O\)�*=q��=qC޸��\)@p  ����O�
C��                                    Bxg�}�  �          A���@Tz��/\)��33C
���@vff��Q��V�HC�                                     Bxg��4  
�          A�R��\)@X���'
=���HC޸��\)@xQ��ff�EC�                                     Bxg���  �          A
=���@(Q�������C!����@E���Q��9p�C�                                    Bxg���  �          A  ��G�@=p��$z�����C�{��G�@\�Ϳ����IG�Cff                                    Bxg��&  f          A  ��=q@<(��\)���\C� ��=q@Z�H���
�Ap�C��                                    Bxg���  �          A���@6ff�%����CT{��@Vff����L��C{                                    Bxg��r  �          Aff��@8Q��(���p�C���@W
=�޸R�?�C޸                                    Bxg��  �          Aff��33@7��6ff���\C�f��33@\(��Q��k�
C�3                                    Bxg��  �          A{��Q�@@  �9����(�C�
��Q�@dz��
=q�o33C�
                                    Bxg�d  �          Aff��@E�%���C5���@e��=q�IC�q                                    Bxg�
  �          A�H�陚@L(��-p����\Cff�陚@mp���
=�Tz�C                                    Bxg��  �          A=q��=q@5�^{���C���=q@b�\�0  ��33C^�                                    Bxg�-V  �          A���@n�R�p��x  C.���@�(�������HC��                                    Bxg�;�  �          A���(�@S33�J=q��  CY���(�@{�����CE                                    Bxg�J�  �          A����H@N�R�
=q�qp�CB����H@hQ쿰����C��                                    Bxg�YH  �          A���@E�*�H���C޸��@g���z��T  CxR                                    Bxg�g�  T          Ap���33@\)����  C!޸��33@=p���  �Ap�C�R                                    Bxg�v�  �          A����@_\)�
=��=qC�3����@|(���G��'�
C(�                                    Bxg��:  �          A�����@ff�z��j�\C"������@0�׿�p��'
=C޸                                    Bxg���  �          A����33@�H>��
@��C"�)��33@�?\(�@�G�C#�                                    Bxg���  �          A��� z�?���?�
=A
=C*=q� z�?��\?ٙ�A>=qC,�                                     Bxg��,  �          Ap����?\@�A~=qC)����?xQ�@$z�A��C,��                                    Bxg���  �          A�\��Q�?�33?��A1G�C)��Q�?�G�?���AQ�C,�\                                    Bxg��x  �          A\)��?�Q�˅�-C&}q��@  ��{����C$h�                                    Bxg��  T          A��Q�?��ÿ^�R���
C):��Q�?�p����QG�C(0�                                    Bxg���  �          A\)� ��@�ͽu��
=C$�� ��@
=q>�G�@C33C$�R                                    Bxg��j  �          A����@p  ?�@w
=C�f���@aG�?�z�A�C�                                    Bxg�	  �          A  ��33@n{?333@�\)C���33@]p�?��
A&�RC��                                    Bxg��  �          A\)� ��@"�\>���@�C"xR� ��@��?aG�@�ffC#\)                                    Bxg�&\  �          A���H?8Q�>\@$z�C/#���H?(�?�@g�C/�)                                    Bxg�5  �          A
=�=q?�R?��@��C/Ǯ�=q>�?:�H@�
=C0ٚ                                    Bxg�C�  �          A���H?�R?�@dz�C/�\��H>�?(��@�{C0�                                     Bxg�RN  �          A33�=q?O\)?(�@�z�C.� �=q?&ff?G�@���C/��                                    Bxg�`�  �          A����?�ff?��
@�\)C+{���?��
?�ffA��C,�                                    Bxg�o�  �          A33� ��@(�?�@�ffC$�=� ��?�{?��A3�
C'                                      Bxg�~@  �          A  ��ff@%�?��RA!�C"���ff@
=q@33A_�C$�\                                    Bxg���  �          A����=q@O\)?�  A{C���=q@7
=?�(�AT��C�                                    Bxg���  
�          AQ����\@>�R?�Q�A�C+����\@#�
@Ad  C!��                                    Bxg��2  �          Az���33@q�?O\)@�
=C����33@^�R?�A5�C^�                                    Bxg���  �          A�
��\@~�R?��A=qC����\@c33@\)Au��C#�                                    Bxg��~  �          A(�����@��\?u@�\)C)����@o\)?��AM��C+�                                    Bxg��$  �          Az����H@��>��?�p�C�
���H@�(�?�G�AQ�C��                                    Bxg���  �          A(����
@���>#�
?�ffC޸���
@�\)?�  A�RC��                                    Bxg��p  �          A\)��=q@�33>u?��C���=q@��?���A  C                                    Bxg�  �          A�ə�@�p�@(�At��C  �ə�@�Q�@R�\A���C�f                                    Bxg��  �          AG����H@���?u@�33C�\���H@���@	��AxQ�C�                                    Bxg�b  �          A\)��\)@��\@Y��A�\)CǮ��\)@��@��
B�C	5�                                    Bxg�.  �          AG��~{@���@��RA��B���~{@���@���B)�C
                                    Bxg�<�  �          A�R��(�@���@G�A��C����(�@�  @�(�B��C��                                    Bxg�KT  T          A\)��(�@�G�?�{@�Q�C�R��(�@�(�@�A��HC�                                    Bxg�Y�  �          A  ��{@�Q�?���A(�C����{@�G�@"�\A��CB�                                    Bxg�h�  �          Az���G�@��
�B�\���CǮ��G�@���?k�@�{CL�                                    Bxg�wF  �          Az���@�\)������C����@�������HC��                                    Bxg���  �          Az���Q�@���?��HA_33C����Q�@��@B�\A�Q�C^�                                    Bxg���  �          A����z�@���?p��@���C���z�@���@	��AqG�C:�                                    Bxg��8  �          A������@�  @0  A��C� ����@�p�@��A���C�R                                    Bxg���  �          A����p�@��
@g�AΣ�C8R��p�@��\@��B�C	�                                    Bxg���  �          AQ��\@��@��Ap  C�3�\@��@Tz�A�C��                                    Bxg��*  �          Az���33@�{?�ffAJ�RC	\��33@�33@C�
A���C\)                                    Bxg���  �          A����  @�\)?�Q�A[
=C����  @��H@QG�A�z�C	0�                                    Bxg��v  �          A�H���@���@l��A�(�C�R���@�\)@�  Bp�C�=                                    Bxg��  T          A������@�
=@%A�G�C����@��@}p�A݅C�                                    Bxg�	�  
�          A	����R@�@   A�33C����R@�z�@w�A�  C
�                                    Bxg�h  �          A
�\��=q@���@	��Af{C����=q@�=q@g�Ař�C�                                    Bxg�'  �          A
=q��=q@���@'�A��C����=q@�
=@���A�Q�C�
                                    Bxg�5�  �          A
=q����@ȣ�@  Aqp�CW
����@���@n{A��
C�3                                    Bxg�DZ  �          A	����Q�@��H@?\)A���C�)��Q�@�p�@�z�A�{C�=                                    Bxg�S   �          A
�H�Ϯ@�z�?��@���C
���Ϯ@�{@\)A��CW
                                    Bxg�a�  �          A
�H��33@�{�   �R�\C5���33@��?=p�@��
C^�                                    Bxg�pL  �          A	���ۅ@�z�=u>��C&f�ۅ@�
=?�=qA�RC�                                    Bxg�~�  �          A	����@��R>�33@z�C�����@��R?У�A.�HC��                                    Bxg���  �          A����\)@���>W
=?�33C�H��\)@�=q?�G�A"�HC
=                                    Bxg��>  �          A�H���
@�
=?5@��CW
���
@�(�@33A`��C0�                                    Bxg���  �          A�\�ȣ�@�=q?!G�@��C
aH�ȣ�@��@   A\  C�                                    Bxg���  T          Aff��Q�@��>W
=?��HC�=��Q�@��H?��
A((�C��                                    Bxg��0  T          A=q����@��>��R@	��C=q����@�  ?�\)A1C�{                                    Bxg���  T          AG��˅@��\?�@z�HC��˅@���?��AQ��C��                                    Bxg��|  �          A�
��{@�(��\)�z�HCp���{@�  ?�33A ��C&f                                    Bxg��"  �          A\)��p�@���L�;���Cp���p�@��R?��RA
=CG�                                    Bxg��  �          Ap�����@�ff���
��C� ����@���?�p�A�CL�                                    Bxg�n  �          Ap����
@�����E�C�3���
@�z�?Y��@���C5�                                    Bxg�   �          AG���33@�  ��p���C  ��33@�
=�k�����C��                                    Bxg�.�  �          A{���
@�=q��z���RC�3���
@��׾���n{C�H                                    Bxg�=`  �          A����
=@�ff��  ��ffCxR��
=@��=L��>�z�C�f                                    Bxg�L  �          A���33@�����33�33Cc���33@����\)���Ch�                                    Bxg�Z�  �          A{��ff@�\)�޸R�0(�C=q��ff@�Q��
=�'�CǮ                                    Bxg�iR  �          A��Q�@�=q���:�\CJ=��Q�@�(�����]p�C��                                    Bxg�w�  �          A33��
=@��ff�S33C����
=@�녿E���G�C��                                    Bxg���  �          AG���@�������t��C����@����\)��Cz�                                    Bxg��D  T          A����ff@�33��R�d��CO\��ff@��׿xQ���{C
=                                    Bxg���  �          A�����@�\)��
�lz�CB�����@�p���=q�ۅC�{                                    Bxg���  �          Aff��(�@�Q���
�i�Ch���(�@�ff�����ָRC                                      Bxg��6  �          A\)���@�\)��
=�A��CW
���@�녿#�
��Q�C��                                    Bxg���  �          A
=��\)@�ff�
�H�[
=C{��\)@���s33��p�C�
                                    Bxg�ނ  �          A\)��{@�ff��z��?�
C�=��{@�G��!G��z=qC�                                     Bxg��(  �          A���
=@�ff��\�0��C�f��
=@�  ���@��C�                                    Bxg���  �          Az����
@�ff��33�<��C  ���
@��׿
=q�VffCT{                                    Bxg�
t  �          A(���@�녿�z��>�\C����@�z�
=�j=qC5�                                    Bxg�  �          A�R��@�
=��p��G33C+���@��\�+����RCJ=                                    Bxg�'�  �          A�\��@�ff���H�F{CO\��@����(�����Cp�                                    Bxg�6f  �          A�R��@����Q��C�C���@��R�(����z�C&f                                    Bxg�E  �          A�H��@�p���\)�;�
C����@�  ���eC�3                                    Bxg�S�  �          A\)��{@����ff�4z�C^���{@�G����H�B�\C�q                                    Bxg�bX  �          A�H��ff@�������6�HC�
��ff@�
=���R�\C&f                                    Bxg�p�  �          A
=��\)@��H��33�?
=CB���\)@���R�vffCp�                                    Bxg��  �          A�\��\)@�  �����D��CǮ��\)@���0����=qCٚ                                    Bxg��J  �          A�\��  @��׿����:�\C��  @�33�z��j=qC�R                                    Bxg���  �          A=q���R@��R���H�F�HC�����R@�=q�5��{C�q                                    Bxg���  �          A���H@�p���Q��D��CaH���H@��׿�R�{�C��                                    Bxg��<  �          A���  @�G���\�O�C  ��  @��Q���C�
                                    Bxg���  �          A��ff@��ÿ����8  C����ff@�33�
=q�XQ�C�=                                    Bxg�׈  �          A33��@��
��  �4��C(���@����S�
Cn                                    Bxg��.  �          A���Q�@�(���z��(��C����Q�@�p���
=�*�HC��                                    Bxg���  �          A����G�@�G����O33C���G�@�p��:�H���C޸                                    Bxg�z  �          Az���@�
=��\�4��C�q��@��þ�(��.{C�                                    Bxg�   �          A�\�陚@�녿�Q��.�RC�f�陚@��\���
�C(�                                    Bxg� �  �          A33��=q@�����G��6=qC�q��=q@�33�����#33C#�                                    Bxg�/l  �          Az���{@�녿ٙ��.=qC���{@��H������C��                                    Bxg�>  �          A�
��R@�����H�IG�CJ=��R@�
=�!G�����CW
                                    Bxg�L�  �          A\)��\@�(���33�D  C  ��\@�\)�#�
��=qC
=                                    Bxg�[^  �          A33���
@��׿��F�\C�q���
@�(��.{���
C�3                                    Bxg�j  �          Az�����@�33���R�K�C\)����@�\)�8Q����\CE                                    Bxg�x�  �          A�H��33@�
=�z��U�C�3��33@��
�Tz����\C��                                    Bxg��P  �          A  ��
=@��
�ff�W�C����
=@�G��aG���(�C��                                    Bxg���  �          Az���=q@{���R�d��CW
��=q@�zῇ���Q�C�3                                    Bxg���  �          A\)����@s33�p��d��C
=����@�Q쿊=q����C\)                                    Bxg��B  �          A
=��ff@~�R���b=qC���ff@���  ���C)                                    Bxg���  �          A33���
@mp��ff�YG�C�=���
@�z�}p����
C=q                                    Bxg�Ў  �          A���z�@p  � ���O33C�{��z�@���c�
��{C33                                    Bxg��4  �          A���   @n�R��Q��F{C��   @��
�Q���
=C�                                     Bxg���  �          A���{@e��33�(Q�C(��{@z=q�z��l��CQ�                                    Bxg���  �          Az��Q�@R�\��ff�{CO\�Q�@e���g
=C�                                    Bxg�&  �          A�R� ��@XQ��\�6�\CE� ��@o\)�@  ���C!H                                    Bxg��  �          A��@X�ÿ��:{CO\�@p�׿G�����C)                                    Bxg�(r  �          A(��=q@R�\���P��C  �=q@n�R���
�љ�Cc�                                    Bxg�7  �          A  � ��@R�\���j�RC�=� ��@s33��G�� ��C                                    Bxg�E�  �          A  � z�@Vff�33�m��C^�� z�@w
=���\�p�CO\                                    Bxg�Td  �          A  ��ff@^�R�
=�s33Cff��ff@�  ���
��RCL�                                    Bxg�c
  �          A��=q@u���\�Lz�C�=�=q@�  �^�R���Ch�                                    Bxg�q�  	.          A�����@�����Tz�C� ����@�p��Q���(�Cff                                    Bxg��V  �          Az����H@�녿������Cz����H@�ff>aG�?�\)C��                                    Bxg���  �          A����z�@����  �ə�C����z�@�{>�(�@,��C
                                    Bxg���  �          A����G�@�Q쿢�\���\CB���G�@��>W
=?�ffCu�                                    Bxg��H  �          A���{@�(����H�(�C#���{@��\=�\)>�G�C�                                    Bxg���  �          AG����
@���\��CY����
@��R=L��>���CG�                                    Bxg�ɔ  �          AG�� (�@���	���U�C!H� (�@��W
=��ffC�R                                    Bxg��:  T          A�
��\)@������=G�C  ��\)@���
=�j�HC�q                                    Bxg���  �          A��\)@�
=��\)���HC)��\)@��
>\)?h��CO\                                    Bxg���  �          A=q��@~�R��ff�  C�q��@�Q쾨����
C.                                    Bxg�,  �          A33� ��@\)���H�EG�C�{� ��@�z�8Q����CY�                                    Bxg��  �          A33��\)@�����F=qC�3��\)@����
=�c�
C��                                    Bxg�!x  T          A{�   @�\)�����;
=C!H�   @��\��
=�{CY�                                    Bxg�0  T          A���{@�z�����<Q�CB��{@�Q��\�C�
CW
                                    Bxg�>�  �          A���33@��
����5�C���33@�
=���/\)C�=                                    Bxg�Mj  �          A  �p�@�z���R�<��C�f�p�@��׿���Mp�C�                                    Bxg�\  �          A�
���@����H�:{C�
���@����   �<��C��                                    Bxg�j�  �          Ap��@������Lz�C(��@�=q�(���|(�C��                                    Bxg�y\  �          Aff�@���(��Q��C�R�@�p��0�����C}q                                    Bxg��  T          A{�
=@�p���Q��:ffCW
�
=@��þ��6ffCs3                                    Bxg���  �          A\)�33@���R�T(�CE�33@�(��=p����C��                                    Bxg��N  �          A�H�=q@�z�����e�CO\�=q@��Ϳk���C��                                    Bxg���  �          A�� ��@�=q�   �n{C!H� ��@��H�s33���Cu�                                    Bxg�  �          A��   @��H�(Q��{\)C���   @���������C��                                    Bxg��@  �          A�H� (�@�
=�+����HC��� (�@�����33��33C��                                    Bxg���  �          A{� ��@�ff�{�mC��� ��@�
=�s33��p�C�q                                    Bxg��  �          A  �@�G��!G��o33Ch��@��\�u���C��                                    Bxg��2  �          A���G�@�(��%�uG�C޸�G�@���  ��(�C
                                    Bxg��  T          A���\)@�\)�   �l��C\�\)@�Q�xQ�����CW
                                    Bxg�~  �          AQ��\)@�{����hQ�C@ �\)@��R�k�����C�
                                    Bxg�)$  �          A����@�\)�%��v=qC�����@��ÿ��
���HC޸                                    Bxg�7�  �          A=q���@�33�   �pQ�CaH���@�z�}p���(�C��                                    Bxg�Fp  �          A
=��{@����%��}��C���{@�
=��\)�ڏ\C�3                                    Bxg�U  �          AQ����@�{�)����G�C����@�G������HCǮ                                    Bxg�c�  �          AQ���{@�  �+����RCu���{@�33��
=��CG�                                    Bxg�rb  �          Az����@���*�H��  C���@�ff�����ڏ\C�f                                    Bxg��  �          A  ��z�@����0������C+���z�@��Ϳ��R��C޸                                    Bxg���  �          A�
���
@��:�H���C�����
@�����	G�C��                                    Bxg��T  T          A�� ��@�33�>�R��ffC�H� ��@�����  ��C��                                    Bxg���  T          A{���H@��@  ��G�C�3���H@��Ϳ����{C�                                    Bxg���  �          A{���\@�  �<(���  C� ���\@�ff��p����C�3                                    Bxg��F  �          A���{@���<(���=qC���{@�  �����(�C�                                    Bxg���  �          A��{@�Q��@  ���Cn��{@��R���H��C��                                    Bxg��  �          A�R�G�@�ff�O\)����C� �G�@�  ���
�)G�C:�                                    Bxg��8  �          A�H� Q�@��\�N�R��(�C�H� Q�@��
��p��$Q�Cn                                    Bxg��  �          A(��G�@�G��XQ���(�C��G�@�(���\)�1G�C�)                                    Bxg��  �          A�
�=q@g
=�|�����C
�=q@���'��yC5�                                    Bxg�"*  �          A=q���@fff�q����HC����@�33�{�mG�Cc�                                    Bxg�0�  �          A��p�@hQ��e��ffCٚ�p�@�=q�G��\  C�=                                    Bxg�?v  �          A��@x���Z=q��(�CaH�@�Q��G��A��C��                                    Bxg�N  �          A���p�@�33�HQ���=qC!H�p�@�(���
=�!G�C�                                    Bxg�\�  �          Ap��{@�G��E��ffC�{�{@�녿�z��\)Cff                                    Bxg�kh  �          A����@\)�4z���33C!H��@���z��  CaH                                    Bxg�z  �          Az��Q�@n{�<����=qC�q�Q�@�
=��\)�z�C�)                                    Bxg���  �          A����@u��>�R����C���@��\��{��C�                                    Bxg��Z  �          A���33@����;���=qC���33@�Q쿾�R�\)C�H                                    Bxg��   �          A�\��@���,(���G�C����@�  ���R���
CB�                                    Bxg���  �          A33���@����,(����HC�{���@�G���p����C&f                                    Bxg��L  �          A���Q�@����p��h��C
=�Q�@�
=��  ��z�C�                                    Bxg���  �          A�
��@�(��p��j�\C���@�ff��G���ffC��                                    Bxg���  �          A��{@������g\)C��{@��\�k���ffC�                                    Bxg��>  �          A  ���@�(���R�k\)CQ����@�ff�n{��  C^�                                    Bxg���  �          Ap��@��H�5��z�C�{�@��׿�����C                                      Bxg��  �          A��p�@��R�2�\��p�C�
�p�@�(�������
=Cn                                    Bxg�0  �          A=q�33@���/\)����C���33@�Q쿗
=�ۅCY�                                    Bxg�)�  �          A�\��\@���.�R�\)C���\@�(���\)��  C��                                    Bxg�8|  �          AG����@�{�7���  C�����@�(����\��ffC:�                                    Bxg�G"  �          A\)� Q�@���R�\��=qC33� Q�@����
=��RC�
                                    Bxg�U�  �          A\)��
=@�z����H����C�\��
=@�\)���t��C��                                    Bxg�dn  �          A���  @�
=��ff��G�C����  @��H� ���33CY�                                    Bxg�s  �          A���G�@����G���ffC����G�@����%���z�C:�                                    Bxg���  �          A(���Q�@�(������{C�
��Q�@�G��#33��{Ck�                                    Bxg��`  �          A����Q�@��\�����
=C����Q�@����*=q��33C}q                                    Bxg��  �          Az����H@��
�{����HCn���H@�p��G��i��C�
                                    Bxg���  �          A���߮@��H�z=q�˙�C+��߮@�z�����f{C^�                                    Bxg��R  �          Aff��{@��\�~{���
C���{@�����
�p��C{                                    Bxg���  �          A  ��Q�@��\�{���G�CaH��Q�@�z��G��pQ�Cff                                    Bxg�ٞ  �          A�
����@�ff������{C0�����@�33�!G���=qC�q                                    Bxg��D  �          A  ���@��\�~�R���C�����@����
�m�C��                                    Bxg���  �          AQ���=q@����z=q��=qC���=q@��H����o�C�f                                    Bxg��  �          A
=��
=@��R�~�R��=qC���
=@�����z�\C�                                    Bxg�6  �          A
�H��
=@�  �{���G�C����
=@�=q���s\)C�=                                    Bxg�"�  �          AQ���p�@�\)�|(���p�C)��p�@�G�����h��C8R                                    Bxg�1�  �          A=q��(�@�33������z�C8R��(�@�
=�
=�vffC
�                                    Bxg�@(  �          A���
=@�=q��\)��RC�=��
=@�
=�(��|z�C
��                                    Bxg�N�  �          A=q��33@���������(�C�=��33@�ff�   ����C
{                                    Bxg�]t  �          A����p�@��R���
��ffC:���p�@��H���y�C
��                                    Bxg�l  �          A�����@�p��|(����HC\)����@�  �p��j�RCY�                                    Bxg�z�  �          A(��˅@�����(����CL��˅@�G�����|��C
�                                    Bxg��f  �          A(����
@��\��p���33C�����
@�������CG�                                    Bxg��  �          AQ��ȣ�@���Q���{C���ȣ�@��
�\)���C
!H                                    Bxg���  �          A��ʏ\@�������  C@ �ʏ\@�G��%��z�C	��                                    Bxg��X  �          A�H�ʏ\@����{��{C��ʏ\@�\)�(Q����C	�
                                    Bxg���  �          A�H���
@�  ��33��p�C����
@�
=�#33����C
{                                    Bxg�Ҥ  �          A����H@�33���H�܏\C�q���H@�\)����m�C	�H                                    Bxg��J  �          A�����
@�����  �أ�CxR���
@����(��g
=C
p�                                    Bxg���  �          A����p�@�G������RC����p�@�  ��R���HC	�                                    Bxg���  �          A����@�p���G����
C����@��
��H���\C��                                    Bxg�<  �          A
ff�ƸR@�=q�z=q��\)C�)�ƸR@�z���_�
C	��                                    Bxg��  �          A
�R��=q@���~{��Q�C
=��=q@�=q��^�RC5�                                    Bxg�*�  T          A
{����@�\)�~{��33C�
����@�=q���_
=C                                      Bxg�9.  T          A
{��Q�@��������C
��Q�@�G��	���g33C\                                    Bxg�G�  �          A
=��ff@���������ffC=q��ff@�p��{�m��C�                                    Bxg�Vz  �          AQ�����@�������G�C�����@�p��z��uG�C��                                    Bxg�e   �          A
�H���@�
=����޸RC����@��H�	���d��C                                    Bxg�s�  �          AQ��ƸR@��R�����{Cٚ�ƸR@��\�	���b�RC�\                                    Bxg��l  �          A��\)@�\)��G���
=C޸��\)@��ff�p  C	��                                    Bxg��  �          A����
=@�����
����CaH��
=@�ff�
�H�]C	aH                                    Bxg���  �          A�
����@�ff��  ��33CB�����@�����V=qC
\)                                    Bxg��^  �          A���Ϯ@�  ��(���C�=�Ϯ@����(��`Q�C	�3                                    Bxg��  �          A
=�˅@������\��\)C
�˅@���Q��\(�C	{                                    Bxg�˪  �          AQ���{@�z�������  CJ=��{@Ǯ��(��I�C�                                     Bxg��P  �          A��Ǯ@����  ��Q�C�3�Ǯ@�G��
=q�[33C��                                    Bxg���  �          A�\��  @�\)��
=���
Ck���  @�  �=q�s�C�q                                    Bxg���  T          A�R��33@��������CW
��33@�
=�  �c\)CQ�                                    Bxg�B  �          A33��@�\)��33���C	G���@�p��
=�T��C��                                    Bxg��  �          A���33@�=q��33��33C�R��33@�(���z��%��C &f                                    Bxg�#�  �          A���z�@�G���=q���C�{��z�@�\����B���                                    Bxg�24  �          AQ����@�33���H��p�CO\���@����H�*�\Cz�                                    Bxg�@�  �          Ap����H@��������  C
����H@��
��ff�1p�C�                                    Bxg�O�  �          A���(�@���������z�C0���(�@�p���
�L(�CG�                                    Bxg�^&  �          A=q�ڏ\@���w���33C0��ڏ\@����G��-p�C	�                                    Bxg�l�  �          A�
��R@r�\�x���ǅC
��R@�{����b�\C�                                     Bxg�{r  �          A������@P���l����  C������@�(��G��g
=C33                                    Bxg��  �          A����
=@Tz��tz��ͅCL���
=@�\)�
=�x  C(�                                    Bxg���  �          A  ��(�@@  �������C#���(�@����+�����C\                                    Bxg��d  �          A33��
=@:=q�y�����HC���
=@�(��$z����C=q                                    Bxg��
  
�          A	��\)@0���qG����C#���\)@|(�� ����p�Cff                                    Bxg�İ  �          A
{��Q�@(Q��u�ә�C ���Q�@w
=�'
=��
=C�                                    Bxg��V  �          A����@=p���Q���
=C�����@�\)�)����G�Cp�                                    Bxg���  �          A33�ᙚ@<��������{CQ��ᙚ@����1����C�{                                    Bxg��  �          A�����@���������{C�����@�33��H�y��C	��                                    Bxg��H  �          A�\����@���������\)C�����@�33� ����
=C33                                    Bxg��  �          A���(�@Fff�s33��C����(�@����������C&f                                    Bxg��  �          A=q��Q�@z���\)���C ff��Q�@n{�Dz���{CG�                                    Bxg�+:  �          A�R��{@���~�R��=qC :���{@mp��3�
��p�C�q                                    Bxg�9�  �          Aff�ٙ�?��H��������C#���ٙ�@S�
�@  ��G�C�                                    Bxg�H�  �          A����@���z����
C#aH���@[��E��z�CL�                                    Bxg�W,  �          A  ��(�@������{C#���(�@\(��=p���z�Cu�                                    Bxg�e�  T          A\)���H?�G������C%�=���H@L���N{����C�3                                    Bxg�tx  �          AG�����?�(���Q���=qC%}q����@L(��S�
���Cff                                    Bxg��  �          A ����33?�z���G��{C%����33@H���W
=�ř�C��                                    Bxg���  �          A����@�p��n�R����C����@��R��{�C�B�8R                                    Bxg��j  �          A
=q�%@�{�>{����B�.�%A�
>\)?fffB��)                                    Bxg��  �          A(�����@���~{���CL�����@�G�����RB��3                                    Bxg���  �          A
�H���@�z���33��G�C&f���@�p����iC�                                    Bxg��\  �          A33��@�
=�z�H��
=C����@�Q쿰���ffB�33                                    Bxg��  �          A���z�@�33�fff�îB�B���z�@�
=�W
=����B��)                                    Bxg��  �          A
�\�a�@�
=�N{���RB�{�a�@��;�  ��\)B��                                    Bxg��N  �          A
=�Q�@�=q�U��
=B�B��Q�A �׾��
��B�\)                                    Bxg��  �          A
�H�Vff@�(��Fff���B����VffA   �u���B�ff                                    Bxg��  �          A\)��(�@�G��j=q���HCxR��(�@У׿��
��HC�{                                    Bxg�$@  T          A���(�@����{��G�C
�q��(�@��
��33�F=qC��                                    Bxg�2�  �          Ap���(�@��R������\)C
s3��(�@˅�޸R�5p�C�                                    Bxg�A�  �          A���=q@�33�z=q����C	ff��=q@�{�����$  C�                                    Bxg�P2  �          AQ���ff@����x���ң�C�=��ff@�\)�\��CW
                                    Bxg�^�  �          Az��Å@�Q��s�
��(�C���Å@��H�����'�
C{                                    Bxg�m~  �          A�����H@�p��\)���C\���H@\����;�
C\                                    Bxg�|$  �          Az���G�@���~�R��G�Ck���G�@�z��  �7\)C�                                    Bxg���  �          A���
@�(�������G�Cc����
@Å��Q��JffC�                                    Bxg��p  �          A����@�p��qG���  C	=q���@�ff������C=q                                    Bxg��  �          A�����
@�ff�g��\C�����
@���������
C(�                                    Bxg���  �          AG���
=@�ff�e���\)C{��
=@�zΉ����HC��                                    Bxg��b  �          A�����@�ff��  �׮C
������@��H���.{C��                                    Bxg��  �          A=q���@��R��ff��C�3���@ƸR��Q��ICT{                                    Bxg��  
�          A�H��p�@�  ������
Cc���p�@\�
=q�`(�Cp�                                    Bxg��T  T          A33�Å@��\������C���Å@�p��
�H�`  C�q                                    Bxg���  �          A�R���
@�(��~�R�ӮC
=���
@��ÿ�33�*�RCJ=                                    Bxg��  T          A=q��=q@�z��|����33C�R��=q@��ÿ�\)�(  C�                                    Bxg�F  �          A\)��
=@���������{C���
=@�ff�޸R�3
=C�                                    Bxg�+�  �          A  ���
@�ff��  ����C(����
@��
��p��1p�C#�                                    Bxg�:�  �          A���=q@�(�������G�CO\��=q@�p�� ���L��C�                                    Bxg�I8  T          A=q�ȣ�@���ff��p�C�
�ȣ�@����	���YC�                                    Bxg�W�  T          Aff���@����ff����C!H���@Ǯ����?33C�                                    Bxg�f�  T          A�R�љ�@�  ������C��љ�@�=q�z��Pz�C	.                                    Bxg�u*  
�          A
=��
=@�{������p�C����
=@�\)���H�Ep�C�                                    Bxg���  �          A
=��z�@�p���
=��
=C���z�@У���
�N�RCJ=                                    Bxg��v  �          A�
�ə�@�Q�������
=C� �ə�@����
�H�Y�C��                                    Bxg��  T          A\)�ʏ\@�ff��\)��G�C���ʏ\@ʏ\����W\)C�                                    Bxg���  T          A�H��(�@�G���z����HC.��(�@�G�����>ffC	��                                    Bxg��h  �          A33��ff@��R��ff��33C�f��ff@����(��EC
.                                    Bxg��  T          A\)����@����ff����C�=����@��R�  �b=qC
!H                                    Bxg�۴  �          Az���33@�������C����33@�Q��#33�~�HC	�                                    Bxg��Z  �          Az�����@����\)��33Cs3����@�{�#�
��Q�C
@                                     Bxg��   T          A������@�Q�������HCc�����@��33�e�C
�{                                    Bxg��  �          A���\)@�=q�����C@ ��\)@�\)�33�d(�C�H                                    Bxg�L  T          A����p�@�����\)��\C�=��p�@��H�,����G�C
=                                    Bxg�$�  �          Az���ff@����������HC���ff@���(Q����CQ�                                    Bxg�3�  �          A  ���@|(�������C�����@�ff�$z���33C5�                                    Bxg�B>  �          A  ��z�@x��������=qC�\��z�@�  �4z����Cff                                    Bxg�P�  T          AQ��ۅ@z=q��33��z�CL��ۅ@�G��6ff���C\                                    Bxg�_�  
�          Az���G�@y����\)���C���G�@��H�=p�����C��                                    Bxg�n0  �          A����  @w
=����=qC33��  @���Fff��p�CB�                                    Bxg�|�  �          A���Ӆ@x������	G�C���Ӆ@�{�L����
=CJ=                                    Bxg��|  �          A����@u���R�G�C���@�z��L(���z�C�)                                    Bxg��"  �          A�����
@w
=�����
�RC����
@��P����  CT{                                    Bxg���  �          A���أ�@hQ���
=��
C���أ�@�ff�Q���p�C.                                    Bxg��n  �          A�H��\)@p����G���
C����\)@�33�Q���C33                                    Bxg��  �          A�H�ٙ�@c33��G��33Cu��ٙ�@����H����  C\                                    Bxg�Ժ  �          A=q����@e�����	
=C������@�z��N{����C��                                    Bxg��`  �          A��G�@w
=���\�p�CxR��G�@��R�QG���z�C	�                                     Bxg��  �          A{���@j�H���
�  C.���@���XQ����C\                                    Bxg� �  �          A�H��ff@g
=��{�=qC� ��ff@�G��]p����CaH                                    Bxg�R  T          A�
��ff@l�������C33��ff@�(��^{��=qC
�H                                    Bxg��  �          A  ��@^�R�����RC� ��@�(��\(����\C!H                                    Bxg�,�  �          A���߮@@�����H�Q�C���߮@�ff�g
=���C��                                    Bxg�;D  �          A����{@H�����\�{C�H��{@�=q�c33��p�C��                                    Bxg�I�  �          A\)�أ�@h�����H��C���أ�@��G���  CB�                                    Bxg�X�  �          A\)�ҏ\@�����  ��HC�)�ҏ\@���8Q���C
�                                    Bxg�g6  �          A
=��G�@|(����H���C����G�@��R�?\)���C
�                                    Bxg�u�  �          A  ��  @r�\��G��\)C����  @����@  ���C��                                    Bxg���  �          A{��(�@{���\)��RC^���(�@���U���p�CQ�                                    Bxg��(  �          AQ���{@��H����Q�C����{@����U���ffC��                                    Bxg���  �          A=q��ff@�����\)��Cٚ��ff@�\)�QG���\)C	+�                                    Bxg��t  �          A{��  @����ff��C����  @\�]p���33C��                                    Bxg��  
�          AG����
@r�\����
CB����
@�=q�c�
��33C	�\                                    Bxg���  �          A��Ϯ@b�\��  ���CaH�Ϯ@�(��n�R����C                                    Bxg��f  �          A\)��(�@\����G��{C���(�@�=q�s33��\)C�R                                    Bxg��  �          A
=��z�@e��p���C�H��z�@����g�����C��                                    Bxg���  �          A=q��{@`������p�CT{��{@����c�
��(�Ck�                                    Bxg�X  �          A�H��  @a������Q�C^���  @����a���  C�\                                    Bxg��  �          A
=��z�@dz���p���
C�q��z�@�z��g�����C��                                    Bxg�%�  �          A�R�љ�@`����Q��33C��љ�@�(��n{����CO\                                    Bxg�4J  �          A��p�@U���z��p�Cz���p�@���l(����C�                                    Bxg�B�  �          A��(�@:�H������RC=q��(�@�����33��{C!H                                    Bxg�Q�  �          A���
@!���  �$�
CW
���
@�33��33��(�C��                                    Bxg�`<  �          A  ���@@  ������C�����@��H�p  �ģ�C�=                                    Bxg�n�  �          A����
=@W���(��33C�\��
=@�ff�i������C
�                                     Bxg�}�  T          A���z�@s�
����  C(���z�@�(��?\)���C��                                    Bxg��.  T          Az��أ�@�  ��� ffCc��أ�@�\)�/\)��G�C�q                                    Bxg���  �          AQ���ff@z�H����G�C�f��ff@�
=�8Q����HC��                                    Bxg��z  �          Az���{@u������C8R��{@��A���Q�C�f                                    Bxg��   �          A����  @q����=qC���  @�z��C33����C�                                    Bxg���  �          AQ��ҏ\@p�����
�=qC:��ҏ\@��R�N{��Q�C�                                    Bxg��l  �          A�H��Q�@S33�����C���Q�@�(��e��{Cp�                                    Bxg��  �          A  ����@aG������C�=����@�p��I�����C^�                                    Bxg��  �          A(���ff@X����p��G�C���ff@���Z�H��33C
                                    Bxg�^  �          A�
��(�@\(����  C�)��(�@�ff�Y������C�
                                    Bxg�  �          A���ڏ\@^�R�����
G�C�ڏ\@��P  ��p�C��                                    Bxg��  �          Ap�����@U�����
=C������@���L(����CE                                    Bxg�-P  T          A���
=@Z�H�������C�)��
=@��
�P  ��Q�CaH                                    Bxg�;�  
�          A����@hQ����H�ffCJ=���@�\)�@  ��(�C��                                    Bxg�J�  �          A���z�@fff��z���HCh���z�@�\)�C33��
=C}q                                    Bxg�YB  �          A���ָR@s�
��(��p�Ck��ָR@�p��<����z�C��                                    Bxg�g�  �          A���33@n�R�������HCh���33@��R�+���Q�Cu�                                    Bxg�v�  �          @����G�@33@qG�A�\)C!�R��G�>B�\@�G�B	  C2B�                                    Bxg��4  �          @����?���@\)B�C"����    @�{B=qC4�                                    Bxg���  �          @�33���R?���@��B�C%���R�O\)@��RB"C<�                                    Bxg���  
�          @�����\?�ff@�z�B0ffC#{���\�^�R@���B7=qC=                                    Bxg��&  T          @�����H?��@���BU�C,c����H��Q�@�ffBD  CMk�                                    Bxg���  �          @�\)�H�ÿ
=@��B�{C>���H���g�@���BNffCe\                                    Bxg��r  �          @�
=�4z����@�p�B���C<&f�4z��aG�@�  BX�CgO\                                    Bxg��  �          @����(����@�{B�CCT{�(��tz�@��B]�Cp0�                                    Bxg��  �          @�33�\)�aG�@���B��C8���\)�^{@�G�Bcz�CjO\                                    Bxg��d  �          @�z���Ϳ���@�(�B��qCU���������\@��
BX33Cy�                                     Bxg�	
  �          @�p�������@�  B�B�Cg�\������\)@��BB�HC}�                                     Bxg��  �          A�R�˅�*�H@���B�8RCo8R�˅��\)@�=qB,ffC~�                                    Bxg�&V  �          A
=�^{���@�
=B}\)CE��^{���@��B?��Cf33                                    Bxg�4�  T          A Q��xQ�\)@�
=Bs�\C6�xQ��N{@�BK(�C[��                                    Bxg�C�  �          @������
?�33@���B%{C(n���
���H@���B$�C@+�                                    Bxg�RH  �          @�33��
=?�  @�B ��C&����
=��@��B�C8�
                                    Bxg�`�  T          @�33����?���@�Q�B\)C(L����Ϳ8Q�@���B	p�C:aH                                    Bxg�o�  T          @��\���
?�@O\)A��
C%���
>L��@n{A�\)C2Y�                                    Bxg�~:  �          @�G���z�?�(�@�  B��C)(���z�O\)@�33B�C;5�                                    Bxg���  �          @�=q��(�?G�@�G�B+�C,���(���@�=qB#Q�CD��                                    Bxg���  �          @��H��(�?+�@��
B8{C,�)��(���@��\B+��CG�)                                    Bxg��,  T          @����=q?�@��HB	��C/G���=q��p�@��Bp�CA5�                                    Bxg���  �          @��H���?�p�@�\)B�C&�\��녿z�@�ffBG�C9G�                                    Bxg��x  �          @����ə�?�ff@���BG�C$\�ə����@�Q�B\)C6Y�                                    Bxg��  �          @��
���R?=p�@��\B?C+�����R��(�@�G�B3ffCH�                                    Bxg���  >          @��
���?�\@�z�BOffC������Q�@���BV  CB�f                                    Bxg��j  	�          @��R��p���z�@���BJQ�Cy� ��p��ۅ@Q�A���C�\                                    Bxg�  T          @��R�333�z�@�  B�8RC}E�333���R@�z�B6�
C�\)                                    Bxg��  �          @�=q�1G��\)@�B��C?xR�1G��n{@��HBR��CiY�                                    Bxg�\  �          @��
����?��@'�A�=qC%�����>��H@K�A\C0�                                    Bxg�.  
�          A�R����@��@��BEC�����׿G�@�Q�BUC=Y�                                    Bxg�<�  �          A��G�@\��@�{B3��C��G�>��H@�(�B]�C-ٚ                                    Bxg�KN  "          A33��ff@[�@��B)G�CB���ff?\)@�ffBQ33C-��                                    Bxg�Y�  T          Ap���z´p�@���BS��CF�\��z����@���B\)C_�                                    Bxg�h�  
�          A�����\)@�z�BXp�CJ�����{@��Bp�Cb��                                    Bxg�w@  
�          A(���\)?�{@��HBC��C�=��\)����@ȣ�BK\)C@�                                     Bxg���  4          A�H����@E�@���B%\)C�\����>�\)@ƸRBE�
C0��                                    Bxg���  �          Ap���  @�@��BC��C\)��  �L��@�=qBR�
C=�{                                    Bxg��2  �          A�\����?��@�p�B3��C&p����ÿ�(�@���B2�HCBG�                                    Bxg���  �          A����Q�@G�@���BL��CE��Q�Y��@��
B\��C>��                                    Bxg��~  "          A z�����@/\)@�
=BAQ�C���������@�=qB\��C7�                                    Bxg��$  �          A���p�?��R@��B>��C#���p����@�B?��CC                                    Bxg���  
          AG���\)>�G�@��B'33C/Ǯ��\)�z�@�
=Bz�CG)                                    Bxg��p  
p          @�����
?��\@��
B!�C'Ǯ���
��p�@�z�B"Q�C?��                                    Bxg��  �          @�p�����?��@�  B<�HC$#����Ϳ�ff@���B=�CB�
                                    Bxg�	�  	�          @����(����@�ffBC9�
��(��3�
@��B
=CM�{                                    Bxg�b  P          @�33��p�@%@��
B0{CG���p��8Q�@��RBI�RC6&f                                    Bxg�'  "          @�����?�p�@��B��
B�#׿����  @�B�z�Ckn                                    Bxg�5�  
�          @��?aG�@A�@陚B�8RB�� ?aG��0��@�(�B���C��                                    Bxg�DT  T          A�\?(��@z�@��B�aHB�L�?(�ÿ��A�\B��\C���                                    Bxg�R�  �          A	G��}p�=��
A33B��\C/(��}p��u@���Bx�
C��                                    Bxg�a�  �          AQ쿼(�>#�
A
{B��=C-�Ϳ�(��w
=@�
=BwffCy&f                                    Bxg�pF  "          A�Ϳ�  ?
=A�HB�aHCh���  �X��@��B��fC}�f                                    Bxg�~�  �          A	녿+�?��A
=B��HB���+��/\)A��B���C�'�                                    Bxg���  
�          A(�����s33Ap�B�L�CXT{�����
=@�{B[\)C~��                                    Bxg��8  
�          A����� ��@�B�\Cl������
=@\B<��C�                                    Bxg���  �          @��\�����H@���B!p�C{
������?�A[\)CE                                    Bxg���  �          @�ff���H��ff@���B2{C����H���@
=A��C�n                                    Bxg��*  
�          @�=q�AG���=q?��HA,��Cq�=�AG�������
�[�Cq0�                                    Bxg���  
�          @�p�������33���iC_�f�����g
=���\���CVG�                                    Bxg��v  
�          @�\�A���@C�
A�33Cq��A���ff=#�
>�Q�Ct�{                                    Bxg��  
�          @�p��+���=q@)��A�
=C�0��+���׿Y����p�C�]q                                    Bxg��  �          @��Ϳ�p��ٙ�?�z�Au�C�� ��p������  �?�C��
                                    Bxg�h  �          @�
=��(���R?�\)Ae��C�1쿜(���  ��p��T  C�8R                                    Bxg�   T          Ap��C33��@r�\A�Q�Ct�{�C33��ff>���@�Cw�R                                    Bxg�.�  �          A���p���ff@~�RA��Cf�3��p���?s33@�G�Cl��                                    Bxg�=Z  
�          A�H�������@�\)B��Cg33�����Q�?�(�A&�HCm�f                                    Bxg�L             A�\�����
=@��HB�RCj�������R?���A��Cp�)                                    Bxg�Z�  j          A�\�x�����\@�p�B!
=Ch�
�x����G�@Q�A�(�Cq{                                    Bxg�iL  �          A���(���Q�@�
=B33C^����(���=q@��A|Q�ChQ�                                    Bxg�w�  T          A���R�\���@�z�B#�Cn=q�R�\��
=@\)A|Q�Cu�=                                    Bxg���  �          A
=�&ff��33@�ffBOQ�Co��&ff��(�@hQ�A�  CyG�                                    Bxg��>  
�          A�׿˅�9��@��B�aHCq.�˅�ȣ�@��B"��C�q                                    Bxg���  �          A	���B�\�vff@�(�B{33C�e�B�\��  @���BQ�C���                                    Bxg���            A
�\�u��{@�Q�BdG�C���u��=q@�p�A�ffC�{                                    Bxg��0  j          A��=�Q����H@���BTG�C���=�Q���ff@c33A�{C�Z�                                    Bxg���  �          A
=>\)���H@�ffBS{C��f>\)���@^�RAÅC���                                    Bxg��|  T          A\)?=p�����@�ffBR��C�&f?=p���(�@`  A�=qC��H                                    Bxg��"  T          A�
��\�!�@޸RB���Cg\��\����@�p�BffCy�                                     Bxg���  "          A�\�qG�>�=q@�
=B|��C/���qG��Tz�@�ffBT��C]ff                                    Bxg�
n  
�          A
=�=q��z�@�\)B���CM�f�=q��=q@�
=BH��Csp�                                    Bxg�  �          A  ��z��
=AffB��3Ca:��z����@��
BBp�C|�                                    Bxg�'�  �          A\)�fff��\A
=B��Cs\�fff��
=@�33BC�\C�P�                                    Bxg�6`  "          A  ��녿��A  B��qCkB������z�@θRBF\)C�f                                    Bxg�E  T          A�\��\)��p�A
=B��C^���\)��  @ӅBO��C~�3                                    Bxg�S�  
�          A�����R��A�B�{CL�����R��{@�ffBa��C~T{                                    Bxg�bR  �          Ap����
=q@���B��CB�)����ff@�=qBY��Cs��                                    Bxg�p�  �          @�{���Ϳ:�H@�33B��qCI�{�������@�33BU
=Cv\                                    Bxg��  T          @�=q��  ����@�z�B�(�Cf���  ��G�@�(�B6�C}xR                                    Bxg��D  T          @�G�?E���G�@�\B�ǮC���?E����R@�p�BF�HC�l�                                    Bxg���  �          @�p�=u��
@�\B���C��=u���R@�\)B4p�C�Z�                                    Bxg���  T          @��H�l(��E�@���B&�\C[�f�l(���Q�@G�A�
=Ch8R                                    Bxg��6  
�          @أ������]p�@g�B�CYs3��������?�G�AO33Cb�
                                    Bxg���  �          @�p��a��(Q�@���B:Q�CX�f�a���G�@3�
A�Q�Ch#�                                    Bxg�ׂ  T          @ə��Fff��@��BN�CX��Fff���@K�A���Cj��                                    Bxg��(  p          @�\)���'
=@��Bdp�Cd�{�����
@\(�B\)Ct�f                                    Bxg���  
�          @׮��ff�o\)@�G�BOffCtJ=��ff���\@4z�AƸRC|�{                                    Bxg�t  T          @�\)����r�\@�z�BF33Co޸�������@*�HA�Q�Cy:�                                    Bxg�            @���L����33@�ffBR{C�t{�L����\)@2�\A��C�Q�                                    Bxg� �            @��;�(����@�
=B���C|�3��(����R@��RBE(�C�<)                                    Bxg�/f  
�          @�\)���H��@�=qB�CDG����H�\(�@���BO=qCpJ=                                    Bxg�>  �          @�(����;��
@��BB�
C7�������7
=@��B�RCS��                                    Bxg�L�  �          @��
�ʏ\�.{@\��A��C:(��ʏ\�G�@,(�A�z�CG�q                                    Bxg�[X  �          @�  ��=q��@%�A�{CG�=��=q�O\)?�33A�CO#�                                    Bxg�i�  �          @�G��Å��?���AX  CHn�Å�0  >��@  CL:�                                    Bxg�x�  �          @���\)��p�>�
=@p  C?5���\)���
�u�Q�C?��                                    Bxg��J  "          @�=q���H��
=����0��CD����H����z�����C=z�                                    Bxg���  p          @�(���G��,(�@�HA�ffCL�3��G��a�?O\)@ٙ�CSc�                                    Bxg���  j          @�(������h��@�B=qCZ�\�������H@A�\)Ce��                                    Bxg��<  �          @�
=�i���G
=@��BKC\c��i����33@i��A�
=Clٚ                                    Bxg���  �          @�\)�Y���}p�@�\)Bq��CD=q�Y���y��@��RB/(�Cd�3                                    Bxg�Ј  
Z          @�ff�=p���Q�@�  Bu�HCU@ �=p����@�\)B�Cn��                                    Bxg��.  �          @�=q�(�ÿ�\)@���By�CWO\�(����(�@�ffB�\CpY�                                    Bxg���  �          @�ff��p��5@�
=Bs�
Ck!H��p���Q�@���B
�CzJ=                                    Bxg��z  �          @����0  �Q�@�BX�Cf{�0  ��Q�@dz�A陚Ct��                                    Bxg�   �          @����R�\�g
=@�\)BFG�Cc���R�\��ff@N�RA�p�Cq�                                    Bxg��  �          @����S33@�=qBc��Cl������33@k�A�  Cy��                                    Bxg�(l  �          @���Q��QG�@�Q�B�CV� ��Q���{@#33A��Ccn                                    Bxg�7  �          @�ff��{�5�@�B2G�CTz���{��  @HQ�A���Cdk�                                    Bxg�E�  �          @�z���?�@�\)B���C!c׿���>{@ʏ\Bx��Cr�)                                    Bxg�T^  �          @�p���\)?#�
@�33B�C�f��\)�333@�G�B~ffCs޸                                    Bxg�c  �          @��
�U?���@�\)Bi�
C��U����@��HBpp�CK^�                                    Bxg�q�  �          @��R�\@'�@�
=BV��Cu��R�\��@ȣ�Bw��C={                                    Bxg��P  �          @陚�s33@
=q@��\BT��Cc��s33�}p�@�z�Be��CB�\                                    Bxg���  
�          @����333>L��@�\)B��C/�q�333�8��@���BW��Ca�R                                    Bxg���  �          @�{�����K�@ҏ\B|  C{T{������Q�@�p�B	G�C��\                                    Bxg��B  �          @���9��?�@���B{(�C��9���ٙ�@�B}G�CRW
                                    Bxg���  �          @���\��?E�@�p�Bx=qC'aH�\���'
=@�B\  CY&f                                    Bxg�Ɏ  �          @��H��
=?J=q@���Bb��C)s3��
=�!�@��BK�RCR�q                                    Bxg��4  �          @��
����?�ff@ÅBM  C!�����Ϳ�@\BK��CG�3                                    Bxg���  �          @����p��@�(�@�Q�B2G�CY��p��?^�R@ڏ\Bq��C&��                                    Bxg���  T          @�ff�dz�@�  @�\)B�RB�z��dz�@�R@ָRBg�C\                                    Bxg�&  �          @��H�*=q@ƸR@z=qA�Q�B�aH�*=q@^{@�
=B_��B�                                    Bxg��  �          @�{�\)@�
=@�Q�B�
Bߨ��\)@S33@���Bj�\B�(�                                    Bxg�!r  �          @���mp�@   @�  BG\)C���mp��:�H@��\BZ�\C?0�                                    Bxg�0  T          @�ff���þ�=q@���B9\)C7
�����6ff@�ffBffCQ�=                                    Bxg�>�  �          @�(��\�����R?��RA�33Cl��\����Q�\(���  Cm��                                    Bxg�Md  �          @�׿���ҏ\@%A�p�C}�����  ���\���C~ٚ                                    Bxg�\
  �          @��
���
��(�@�z�B(�C}.���
����?c�
@�G�C�%                                    Bxg�j�  �          @�33�?\)�l(�@�z�BLffCg��?\)��(�@Q�A�p�Ct                                      Bxg�yV  �          @��
���R�"�\@�B�33Cs�ÿ��R����@�G�B  C���                                    Bxg���  �          @��ÿ�G�?z�@�  B�aHC!Ǯ��G��6ff@���Bu��Cnc�                                    Bxg���  �          @�p�������
@���B0CG  �����@n{A�{C[                                      Bxg��H  �          @�\)���ÿ
=@��\B4=qC:�f�����A�@��B�CS
                                    Bxg���  �          @��
��Q�?ٙ�@�\)B2��C!33��Q쿏\)@�(�B9p�C@�H                                    Bxg�  �          @��H��G���Q�@�ffB<  CF�\��G����@�33A��C\��                                    Bxg��:  �          A���33�˅@���B8Q�CD����33����@��A��CZ�{                                    Bxg���  T          A��ff����@�=qB:G�CGQ���ff��\)@�z�A���C\                                    Bxg��  �          AG���Q��$z�@���BEffCQ����Q���  @}p�A�Q�CeO\                                    Bxg��,  �          A{��(���
@���B$\)CJQ���(�����@\(�A�{C[�\                                    Bxg��  �          @�ff��33>�=q@{�B{C1)��33��p�@b�\B ��CE�f                                    Bxg�x  �          A�H��Q�#�
@�{B��C5p���Q��.�R@�p�B��CK�\                                    Bxg�)  �          A���  ?L��@�=qB*p�C,k���  �@�
=B�CG33                                    Bxg�7�  �          A���˅?�  @�Q�B��C&��˅���H@��HB��C>�                                    Bxg�Fj  �          A���ҏ\?&ff@�ffBp�C.aH�ҏ\��@��B�
CDB�                                    Bxg�U  �          A
=��z�?��R@��B(�C'����zῚ�H@�=qB��C=�3                                    Bxg�c�  �          A��߮?�ff@�p�B
=C)� �߮��(�@��
B�\C?�
                                    Bxg�r\  �          Ap���\)@��H@G�A�C	�
��\)@mp�@�  BG�C�                                    Bxg��  �          A\)���
@�p�@0  A�Q�C�{���
@�z�@�
=BQ�Cc�                                    Bxg���  �          A �����
@�z�@AX  C� ���
@�
=@�\)B\)C�f                                    Bxg��N  �          A���@�p�@S�
A�G�C�3��@��@ƸRBz�Ch�                                    Bxg���  �          A���p�@ʏ\@K�A�G�Ck���p�@xQ�@�p�B�C�)                                    Bxg���  
�          A�
���@ƸR@�(�A�Q�CQ����@S33@�{B3\)C�
                                    Bxg��@  �          AQ���33@�ff@�Q�A\C���33@Vff@��HB*{C33                                    Bxg���  �          AG���ff@�
=@;�A��C
+���ff@z=q@��Bz�C��                                    Bxg��  �          Az���  @ə���z���\C����  @���@
=qAdQ�C�f                                    Bxg��2  �          Aff��@Ӆ���Z�HC���@�\)@7
=A�=qCǮ                                    Bxg��  �          A=q��R@�ff@
=AK�C}q��R@��\@�=qA�G�C�                                     Bxg�~  �          AG���Q�@�z�?�{A�RC}q��Q�@���@���A�G�C0�                                    Bxg�"$  �          Az���(�@�
=@1G�A�{C	�R��(�@qG�@���BQ�Cc�                                    Bxg�0�  �          A����z�@�=q@��A�Q�Cz���z�@E�@���B.��CO\                                    Bxg�?p  �          A
=��  @ȣ�@o\)A��RC	
��  @b�\@�z�B��CY�                                    Bxg�N  �          A�H��=q@ȣ�@0  A�=qC	ff��=q@���@�G�BCk�                                    Bxg�\�  �          Ap����H@��@N�RA�C�����H@b�\@��B
=CxR                                    Bxg�kb  
�          A=q��p�@�@8Q�A�\)C@ ��p�@xQ�@�33B	��C�\                                    Bxg�z  �          A (�����@��@(�Ab{C
�H����@���@��HB�Ch�                                    Bxg���  �          A"=q��@��?���A1p�C	ٚ��@���@�G�A��C\                                    Bxg��T  �          A ����33@��H?�A��C(���33@�(�@�z�A�Q�C�H                                    Bxg���  �          A�\��z�@�=q?��A/
=Cs3��z�@�Q�@�=qA�RC��                                    Bxg���  �          Aff��@�(�@   A;33C����@���@��\A��
CG�                                    Bxg��F  �          Aff��@��?�ffA'33C����@���@��A�z�C�)                                    Bxg���  �          A���(�@�G�?�33A2{C	}q��(�@�@�ffA��HC�q                                    Bxg���  �          A{��33@�=q?�z�A\)C\)��33@���@�Q�A�\)C�                                    Bxg��8  �          A����@��?��
@���C�����@�@��\A�
=CaH                                    Bxg���  �          A���ᙚ@���?��@��RC���ᙚ@�=q@�  A��HCG�                                    Bxg��  �          A����
@��
?�\A(Q�C���
@�=q@�(�A�C�3                                    Bxg�*  T          A(���=q@�\)@�Af{C0���=q@��@�ffB�C��                                    Bxg�)�  T          A�
��Q�@�Q�@4z�A�33C�=��Q�@��\@ÅBC^�                                    Bxg�8v  �          A���@�{@9��A�z�C�
���@�Q�@���B�C�                                    Bxg�G  �          A���@���@L(�A��B��\���@�\)@�B+=qC��                                    Bxg�U�  �          A����{@ٙ�?�A!Ch���{@���@�z�A�\)C\                                    Bxg�dh  �          A����33@�  ?��@���C0���33@�(�@�p�A�Q�C��                                    Bxg�s  �          A�R��
=@θR�z��UC#���
=@��@1�A��\C��                                    Bxg���  �          A(���\)@��/\)��  C  ��\)@�=q>���@
=C�                                    Bxg��Z  �          A  ����@^�R��z��  Cu�����@���Fff����C	�                                    Bxg��   �          Az���=q?��
����X�
C)
��=q@�ff���R���C
��                                    Bxg���  �          A����@   �����;\)C ����@�\)��  ���RC
�                                    Bxg��L  �          A  ��{@�p�?���@ᙚC	J=��{@�  @�33AԸRC��                                    Bxg���  �          A����@�p��\����p�C�f���@�=q>#�
?��B���                                    Bxg�٘  q          A���=q@ə�?��@��C33��=q@�  @��A��C�H                                    Bxg��>  
          A!�����@�33>��H@1G�Cٚ����@�
=@h��A��
C33                                    Bxg���  �          A����@�p�<��
=�G�C�q��@��@I��A���C��                                    Bxg��  "          A���߮@�=q�L������C���߮@�녾�(��0  C}q                                    Bxg�0  T          A�H��Q�@�{��(��ʏ\Cp���Q�@�p���ff��\)C                                    Bxg�"�  �          Aff��{@�������ҏ\C:���{@�=q���R���
C
��                                    Bxg�1|  
�          A
=��ff@��R�y����z�C� ��ff@�  �Tz���{C��                                    Bxg�@"  �          Az���G�@�33�U����Cs3��G�@��
�����&ffC
�                                    Bxg�N�  T          AG���ff@�\)=��
?�C�q��ff@�p�@7
=A�\)C                                    Bxg�]n  
�          A�H��{@���?�\@K�C�{��{@�33@S�
A�CY�                                    Bxg�l  "          Az���{@�{    <��
C�H��{@�@1�A�\)C�                                    Bxg�z�  
�          A=q��{@4z�@`  A�=qC u���{?&ff@�{A�33C/Y�                                    Bxg��`  
Z          A���{@(��@<��A�Q�C"���{?E�@xQ�A�  C.�q                                    Bxg��  �          Aff�(�@6ff@-p�A�Q�C!���(�?���@q�A�=qC,�
                                    Bxg���  "          A��@+�@dz�A�33C"u���>�@�{A��HC0�q                                    Bxg��R  T          A�H��@L(�?�Q�A:ffC 5���?�G�@S33A�ffC(��                                    Bxg���  "          A�\�
=@E?��HA�
C ��
=?�@5�A�p�C'�f                                    Bxg�Ҟ  �          A��
ff@N{?xQ�@�Q�C�{�
ff@G�@�RAw�C%E                                    Bxg��D  
�          A��
=@�  ��
=�\)C:��
=@�33?��@�ffC��                                    Bxg���  
�          A  ��\@��\��R�p  C����\@�G�?�
=A#
=CQ�                                    Bxg���  
�          A���\@k���{���CaH��\@n�R?k�@��RC�                                    Bxg�6  
�          A�R��\@1�=L��>��
C"����\@�?�(�Az�C%!H                                    Bxg��  
Z          A�����\@hQ��5���G�C)���\@�=q�z��j�HC�q                                    Bxg�*�  
�          A�����@�33���G�C�q����@�
=?�  @�33C
                                    Bxg�9(  "          A�R���@�  ���� Q�C�����@�
=?�A33C�3                                    Bxg�G�  
�          A��   @��þB�\����C�{�   @�p�@=qAl  C\                                    Bxg�Vt  "          A���G�@��\>��@\)C�)��G�@��
@Dz�A��
C�                                    Bxg�e  T          AG��p�@#33�.{����C#���p�@e����
��C�                                    Bxg�s�  �          A33�  @��S�
���RC$���  @l�Ϳ�33�!�C}q                                    Bxg��f  �          A���(�?�{���\��  C*�f�(�@P  �4z�����C�                                     Bxg��  T          A\)��(�?�{��=q����C,  ��(�@R�\�W
=���C^�                                    Bxg���  �          Az���ff���H���\���C?xR��ff?�(����\�C(�                                     Bxg��X  �          A\)��G��Tz������CN���G�>u�����5�HC1��                                    Bxg���  �          A�\��ff���H�Å�"�RC^���ff�����
=�^
=C?Q�                                    Bxg�ˤ  �          A&=q�˅��{��ff�0�C7�˅@8Q���  ���C�f                                    Bxg��J  
�          A,���	p�@�G���(��˙�C!H�	p�@ȣ׿�z����C��                                    Bxg���  
Z          A0���
=@��H�����z�C�f�
=@��
��z��
�\C�R                                    Bxg���  �          A-���@�\)������
=Cff��@љ��\)�@��CxR                                    Bxg�<  
�          A,�����@o\)�����݅C�R���@����=q�MC��                                    Bxg��  �          A(���(�@N{�������CE�(�@�\)�:=q��z�C:�                                    Bxg�#�  �          A z����R@����G�C#�����R@��
�hQ����C�                                    Bxg�2.  �          AQ���?��H��33�C+\)��@h���qG�����CW
                                    Bxg�@�  "          A
=��R@����=q��
=C$�\��R@�����f�RC��                                    Bxg�Oz  �          A{�G�@l���u����C��G�@��Ϳ�z���RCL�                                    Bxg�^   �          A!���(�@j�H�����p�C�(�@�ff��p��5p�C�\                                    Bxg�l�  �          A�����@`  ����֏\CT{���@����   �=�C��                                    Bxg�{l            Aff��\@(Q���  ��Q�C"����\@���!��lz�C�\                                    Bxg��  7          A�\�{@����
=��Q�C$�)�{@���W
=���RC�R                                    Bxg���  "          A�����@p����\��33C"�����@��\�8����33C�q                                    Bxg��^  
�          A\)��{?�{��Q����C(�f��{@����  ��ffCY�                                    Bxg��  �          A���?�33��(����C'k����@z�H�Vff��Q�C&f                                    Bxg�Ī  
�          Aff��@QG���Tz�CQ���@xQ���J=qC                                    Bxg��P  T          A�
���@j�H��(��\)C.���@y��?(�@r�\C��                                    Bxg���  �          A���
=@U��   �F�RC}q�
=@xQ켣�
��G�CO\                                    Bxg��  �          A���
=?�ff�X����p�C(�q��
=@Dz����_�C�3                                    Bxg��B  �          A���33?����(�����C'G��33@<(���=q�
�\C B�                                    Bxg��  T          A���  @������RC#=q��  @L�ͿO\)����C��                                    Bxg��  "          @�33��Q�?�\)��H��=qC'u���Q�@(Q쿡G���
C �                                    Bxg�+4  �          @��
��{?z��:=q���C/��{?��R�p���Q�C#p�                                    Bxg�9�  �          A���  ��R�����/p�CLL���  ?Tz���  �AffC+s3                                    Bxg�H�  T          A�����
�S�
����1G�CS�f���
>�����\)�Q��C0��                                    Bxg�W&  �          A���z������p��3  CbQ���z�G����p�RC=}q                                    Bxg�e�  T          A����H���\��
=�@(�Cd���H���
�H�}{C:�                                    Bxg�tr  "          A(��e���z���{�?��Clff�e��xQ��
=�HCC
                                    Bxg��  �          A  �n{��  ��Q��{Cm{�n{�����R�n�HCS@                                     Bxg���  T          A  ������  ����+Cf�����������t(�CD��                                    Bxg��d  T          A���E���  ����8�Cp���E������33u�CL\                                    Bxg��
  �          A���w������{�9��Ch���w��p���p���CA��                                    Bxg���  
�          A{�B�\����Ϯ�O{C����B�\��=q�
=¤�RC���                                    Bxg��V  
�          A
=q@��������9(�C�N@�׿��
����C���                                    Bxg���  �          A�\>aG������H�D\)C�>aG�������\ffC��H                                    Bxg��  
�          AQ�?����  ��ff�GffC���?�녿�{��p�C�%                                    Bxg��H  �          A ��?ٙ������
�O�C���?ٙ��������3C���                                    Bxg��  
�          A��?�{�����ҏ\�A��C�L�?�{��33��=qC�}q                                    Bxg��  �          @�������	����=q�=Ca=q���?�  ���fC��                                    Bxg�$:  ?          A  �l(�?�z���R�p\)C��l(�@�=q�����HB�z�                                    Bxg�2�  i          Ap��|(�@������N�C=q�|(�@�ff����Q�B�.                                    Bxg�A�  
�          A��J�H@����  ��p�B�L��J�HA�R�k����B�8R                                    Bxg�P,            A
=���A\)�`  ���B�B����A��?�ff@�RB�=q                                    Bxg�^�  
=          A%����\)@��������B����\)A녿aG����
B�
=                                    Bxg�mx  �          A�R���H@�������߅B��{���H@�z��ff�.{B���                                    Bxg�|  �          A!���=q@�33������p�C�3��=q@�\��{��z�C�f                                    Bxg���  �          A2ff����@�ff�p����
=C�R����A  ?Q�@�{C B�                                    Bxg��j  �          A?���z�Az��[���
=C
��z�A��?\@陚C xR                                    Bxg��  �          A9G���z�A�R��
�8��B�\)��z�A��@#�
AMp�B���                                    Bxg���  T          A7��׮A  ���R� ��B����׮A(�@7
=Ahz�B�(�                                    Bxg��\  "          A(Q��˅@��R����T(�C ���˅A�?�p�A.�HC :�                                    Bxg��  �          A3���ff@�=q������
=C  ��ff@�33���ffCJ=                                    Bxg��  T          A5p����R@��
��  ��\C�
���R@����  �{Cc�                                    Bxg��N  
�          A7\)� ��@��R���\��
=Cn� ��@�
=��\�=qC=q                                    Bxg���  �          A6=q� ��@�33���\��(�C��� ��@��R��p���RC^�                                    Bxg��  "          A$����(�@�Q���(��C��(�@����+��s�C
��                                    